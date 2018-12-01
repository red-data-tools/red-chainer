# frozen_string_literal: true

require 'chainer/functions/activation/relu'

class Chainer::Functions::Activation::ReLUTest < Test::Unit::TestCase

  data(:shape, [[3, 2], []],             keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    # Avoid unstability of numerical grad
    @shape = data[:shape]
    @dtype = data[:dtype]

    @dtype.srand(1) # To avoid false of "nearly_eq().all?", Use fixed seed value.
    @x = @dtype.new(@shape).rand(2) - 1
    @shape.map do |x|
      if (-0.1 < x) and (x < 0.1)
        0.5
      else
        x
      end
    end
    @gy = @dtype.new(@shape).rand(2) - 1
    @check_backward_options = {}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::Relu.relu(x)
    assert_equal(@dtype, y.data.class)
    expected = @x.dup
    if expected.shape == []
      expected[expected < 0] = 0
    else
      @x.each_with_index do |x, *i|
        if x < 0
          expected[*i] = 0
        end
      end
    end
    assert_true(y.data.nearly_eq(expected).all?)
  end

  def test_forward
    check_forward(@x.dup)
  end

  def check_backward(x_data, y_grad, use_cudnn: "always")
    Chainer::check_backward(Chainer::Functions::Activation::Relu.method(:relu), x_data, y_grad, @check_backward_options)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end
end
