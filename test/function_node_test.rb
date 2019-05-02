# frozen_string_literal: true

require 'chainer'

class TestGradTypeCheck < Test::Unit::TestCase
  def test_type_check
    x = Chainer::Variable.new(xm::DFloat.new(2, 3).rand(-1, 1))
    y = x * x
    gx = Chainer::Variable.new(xm::DFloat.new(2, 3).rand(-1, 1))
    gy = Chainer::Variable.new(xm::DFloat.new(2, 3).rand(-1, 1))

    Chainer.grad([y], [x], grad_outputs: [gx], grad_inputs: [gy])

    assert_raise(TypeError) do
      Chainer.grad(y, [x], grad_outputs: [gx], grad_inputs: [gy])
    end
    assert_raise(TypeError) do
      Chainer.grad([y], x, grad_outputs: [gx], grad_inputs: [gy])
    end
    assert_raise(TypeError) do
      Chainer.grad([y], [x], grad_outputs: gx, grad_inputs: [gy])
    end
    assert_raise(TypeError) do
      Chainer.grad([y], [x], grad_outputs: [gx], grad_inputs: gy)
    end
  end
end

class Chainer::GradTestBase < Test::Unit::TestCase
  def setup
    @shape ||= [3]
    @x_names ||= []
    @y_names ||= []

    @xs = init_attrs(@x_names)
    @gxs = init_attrs(to_grad_names(@x_names))
    @gys = init_attrs(to_grad_names(@y_names))

  end

  def forward
    raise NotImplementedError
  end

  def expected_grad
    raise NotImplementedError
  end

  def expected_double_grad
    raise NotImplementedError
  end

  def check_grad
    forward
    ys = @y_names.map { |name| self.instance_variable_get("@#{name}") }
    gxs = Chainer.grad(ys, @xs, grad_outputs: @gys, grad_inputs: @gxs)

    expected = expected_grad
    @gxs.each_with_index do |gx, i|
      expected[i] += gx
    end

    assert_equal(expected.size, gxs.size)
    gxs.zip(expected).each do |a, e|
      Chainer::Testing.assert_allclose(get_value(e), get_value(a))
    end
  end

  def check_double_grad
    forward
    ys = @y_names.map { |name| self.instance_variable_get("@#{name}") }
    gxs = Chainer.grad(ys, @xs, grad_outputs: @gys, grad_inputs: @gxs, enable_double_backprop: true)
    y = gxs.sum
    ggxs = Chainer.grad([y], @xs)

    expected = expected_double_grad
    assert_equal(expected.size, ggxs.size)
    ggxs.zip(expected).each do |a, e|
      Chainer::Testing.assert_allclose(get_value(e), get_value(a))
    end
  end

  private

  def to_grad_names(names)
    names.map { |n| "g#{n}" }
  end

  def init_attrs(names)
    ret = []
    names.each do |name|
      x = xm::DFloat.new(@shape).rand(-4, 6)
      v = Chainer::Variable.new(x, name: name)
      ret << v
      self.instance_variable_set("@#{name}", v)
    end
    ret
  end

  def get_value(x)
    x.is_a?(Chainer::Variable) ? x.data : x
  end
end

class Chainer::TestGradSimple < Chainer::GradTestBase
  def setup
    @x_names = ['x']
    @y_names = ['y']
    super
  end

  def forward
    @y = @x * @x
  end

  def expected_grad
    [2 * @x * @gy]
  end

  def expected_double_grad
    [2 * @gy]
  end

  def test_grad
    check_grad
  end

  def test_double_grad
    check_double_grad
  end
end

class TestGradComplex < Chainer::GradTestBase
  def setup
    @x_names = ['x1', 'x2']
    @y_names = ['y1', 'y2']
    super
  end

  def forward
    @z = @x1 * @x1
    @y1 = @z + @x1 * @x2 + @x2
    @y2 = @z + @y1
  end

  def expected_grad
    dz_dx = 2 * @x1
    dy1_dx = @gy1 + @gy2
    [dy1_dx * (dz_dx + @x2) + @gy2 * dz_dx, dy1_dx * (@x1 + 1)]
  end

  def expected_double_grad
    dy1_dx = @gy1 + @gy2
    [3 * dy1_dx + 2 * @gy2, dy1_dx]
  end

  def test_grad
    check_grad
  end

  def test_double_grad
    check_double_grad
  end
end
