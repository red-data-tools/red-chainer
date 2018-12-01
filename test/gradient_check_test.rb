require 'chainer/gradient_check'

def _uniform(*shape)
  return xm::SFloat.new(shape).rand(2) - 1
end

def _dot(x, y)
  x.zip(y).map{|a| a[0] * a[1]}.inject(:+)
end

class NumericalGradientTest < Test::Unit::TestCase
  def f(xs)
    return [xs[0] ** 2]
  end

  def df(xs)
    return [[2 * xs[0]]]
  end

  def setup()
    @xs = [_uniform(2, 1)]
    @gys = [_uniform(2, 1)]
  end

  def check_numerical_grad_one(f, df, xs, gys, eps)
    dfxs = df.(xs)
    gys = gys.map{|gy| gy.nil? ? 0 : gy}.to_a
    dx_expect = dfxs.map{|dfx| _dot(dfx, gys)}.to_a
    func = lambda do |_|
      return f.call(xs)
    end
    dx_actual = Chainer::numerical_grad(func, xs, gys, eps)
    assert_equal((dx_actual).size, (dx_expect).size)

    dx_expect.zip(dx_actual).each do |e, a|
      Chainer::Testing.assert_allclose(e, a, atol: 1e-3, rtol: 1e-3)
    end
  end

  def check_numerical_grad(f, df, xs, gys, eps=nil)
    if eps.nil?
      eps = [2, 3, 4].map{|i| 10 ** -i}
    else
      if !eps.is_a? Array
        eps = [eps]
      end
    end

    eps.each do |e|
      check_numerical_grad_one(f, df, xs, gys, e)
    end
  end

  def test_numerical_grad()
    check_numerical_grad(method(:f), method(:df), @xs, @gys, @eps)
  end
end

class NumericalGradientReferenceTest < Test::Unit::TestCase
  def setup()
    @x = _uniform(2, 3)
  end

  def check_reference(x)
    # A returned value and an input refers the same memory.
    # See issue https://github.com/chainer/chainer/issues/488
    func = lambda do |x|
      return [x]
    end
    gx, = Chainer::numerical_grad(func, [x], [1])
    Chainer::Testing.assert_allclose(1, gx)
  end

  def test_reference()
    check_reference(@x)
  end
end

class NumericalGradientInvalidEps < NumericalGradientTest
  def check_invalid_eps(xs, gys, eps)
    assert_raise(RuntimeError) {
      check_numerical_grad(method(:f), method(:df), xs, gys, eps)
    }
  end

  def test_numerical_grad()
    check_invalid_eps(@xs, @gys, 0)
    check_invalid_eps(@xs, @gys, -(1.0))
  end
end

class NumericalGradientInvalidType < Test::Unit::TestCase
  def setup()
    @x = xm::NArray.cast(0)
    @y = xm::NArray.cast(0)
    @f = lambda{}
  end

  def test_invalid_inputs()
    y = @y
    assert_raise(ArgumentError) {
      Chainer::numerical_grad(@f, [@x, y], [])
    }
  end

  def test_invalid_outputs()
    y = @y
    assert_raise(NoMethodError) {
      Chainer::numerical_grad(@f, [], [@x, y])
    }
  end

  def test_invalid_mixed()
    y = @y
    assert_raise(ArgumentError) {
      Chainer::numerical_grad(@f, [@x], [y])
    }
  end
end

class NumericalGradientEpsTest < Test::Unit::TestCase
  def setup()
    @x = xm::SFloat.cast(0.0)
    @y = xm::SFloat.cast(1.0)
  end

  def check_different_eps(x, y)
    f = lambda do |x|
      if (-1 < x).all? and (x < 1).all?
        return [x.dup()]
      else
        if (-2 < x).all? and (x < 2).all?
          return [2 * x]
        else
          return [0]
        end
      end
    end
    gx, = Chainer::numerical_grad(f, [x], [y], 0.5)
    assert_equal(1.0, gx)
    gx, = Chainer::numerical_grad(f, [x], [y], 1.5)
    assert_equal(2.0, gx)
    gx, = Chainer::numerical_grad(f, [x], [y], 2.5)
    assert_equal(0.0, gx)
  end

  def test_differenct_eps()
    check_different_eps(@x, @y)
  end
end

class Ident < Chainer::Function
  def forward(inputs)
    return inputs
  end
  def backward(inputs, grads)
    return grads
  end
end

class TestCheckBackward < Test::Unit::TestCase
  data(:dtype, [nil, xm::SFloat, xm::DFloat], keep: true)
  def test_multiple_output()
    @dtype = data[:dtype]
    x1 = xm::DFloat[1]
    x2 = xm::DFloat[1]
    g1 = xm::DFloat[1]
    g2 = xm::DFloat[1]
    f = lambda do |x, y|
      s,t = Ident.new().(x, y)
      u = Ident.new().(t)
      return [s, u]
    end
    Chainer::check_backward(f, [x1, x2], [g1, g2], dtype: @dtype)
  end

  def test_no_grads_for_not_float()
    x1 = xm::DFloat.cast([1])
    x2 = xm::Int32.cast([0, 1])
    g1 = xm::DFloat.cast([1])
    f = lambda do |x, y|
      s = Ident.new().(x)
      return [s]
    end
    Chainer::check_backward(f, [x1, x2], g1)
  end

  def test_no_grads_option()
    x1 = xm::DFloat.cast([1])
    x2 = xm::DFloat.cast([1])
    g1 = xm::DFloat.cast([1])
    f = lambda do |x, y|
      s = Ident.new().(x)
      return [s]
    end

    assert_raise(NoMethodError){Chainer::check_backward(f, [x1, x2], g1, no_grads: [false, false])}
    Chainer::check_backward(f, [x1, x2], g1, no_grads: [false, true])
  end
end
