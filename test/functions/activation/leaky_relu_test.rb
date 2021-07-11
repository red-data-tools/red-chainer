# frozen_string_literal: true

require 'chainer/functions/activation/leaky_relu'

class Chainer::Functions::Activation::LeakyReLUTest < Test::Unit::TestCase

  data(:shape, [[3, 2], []],             keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    # Avoid unstability of numerical grad
    @shape = data[:shape]
    @dtype = data[:dtype]

    @dtype.srand(1) # To avoid false of "nearly_eq.all?", Use fixed seed value.
    @x = @dtype.new(@shape).rand(2) - 1
    @shape.map do |x|
      if (-0.05 < x) and (x < 0.05)
        0.5
      else
        x
      end
    end
    @gy = @dtype.new(@shape).rand(2) - 1
    @ggx = @dtype.new(@shape).rand(2) - 1
    @slope = Random.rand
    @check_forward_options = {}
    @check_backward_options = {}
    @check_backward_options_dtype = xm::DFloat
  end

  def check_forward(x_data)
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::LeakyReLU.leaky_relu(x, slope: @slope)
    assert_equal(@dtype, y.data.class)
    expected = @x.dup
    if expected.shape == []
      expected[expected < 0] *= @slope
    else
      @x.each_with_index do |x, *i|
        if x < 0
          expected[*i] *= @slope
        end
      end
    end
    assert_true(y.data.nearly_eq(expected).all?)
  end

  def test_forward
    check_forward(@x.dup)
  end

  def check_backward(x_data, y_grad)
    func = -> (x) do
      Chainer::Functions::Activation::LeakyReLU.leaky_relu(x, slope: @slope)
    end
    Chainer::check_backward(func, x_data, y_grad, dtype: @check_backward_options_dtype)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad)
    func = -> (x) do
      y = Chainer::Functions::Activation::LeakyReLU.leaky_relu(x, slope: @slope)
      y * y
    end

    Chainer::check_double_backward(func, x_data, y_grad, x_grad_grad, **@check_backward_options)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
