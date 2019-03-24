# frozen_string_literal: true

require 'chainer/functions/array/cast'

class Chainer::Functions::Array::CastTest < Test::Unit::TestCase
  data(:shape, [[3, 4], []], keep: true)
  data(:type, [
    { in_type: xm::SFloat, out_type: xm::DFloat },
    { in_type: xm::DFloat, out_type: xm::SFloat },
  ], keep: true)

  def setup
    @x = data[:type][:in_type].new(*data[:shape]).seq
    @g = data[:type][:out_type].new(*data[:shape]).seq
  end

  def test_forward(data)
    x = Chainer::Variable.new(@x)
    y = Chainer::Functions::Array::Cast.cast(x, data[:type][:out_type])
    assert_equal(y.dtype, data[:type][:out_type])
    assert_equal(y.shape, x.shape)
  end

  def test_backward(data)
    check_backward_options = { eps: 2.0 ** -2, atol: 1e-3, rtol: 1e-2 }
    func = -> x { Chainer::Functions::Array::Cast.cast(x, data[:type][:out_type]) }

    Chainer::check_backward(func, @x, @g, **check_backward_options)
  end
end

class Chainer::Functions::Array::NoCastTest < Test::Unit::TestCase
  def setup
    @dtype = xm::SFloat
    @x = @dtype.new(1).rand
  end

  def test_forward_no_cast_array
    y = Chainer::Functions::Array::Cast.cast(@x, @dtype)
    assert_equal(Chainer::Variable, y.class)
    assert_equal(@x, y.data)
  end

  def test_forward_no_cast_variable
    x = Chainer::Variable.new(@x)
    y = Chainer::Functions::Array::Cast.cast(x, @dtype)
    assert_equal(x, y)
  end
end
