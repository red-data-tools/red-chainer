# frozen_string_literal: true

require 'chainer/functions/activation/sigmoid'

class Chainer::Functions::Activation::SigmoidTest < Test::Unit::TestCase

  data(:shape, [[3, 2], []],             keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @shape = data[:shape]
    @dtype = data[:dtype]
    @x = @dtype.new(@shape).rand(-0.5, 0.5)
    @gy = @dtype.new(@shape).rand(-0.1, 0.1)
    @ggx = @dtype.new(@shape).rand(-1, 1)
    @check_forward_options = {}
    @check_backward_options = {}
    @check_double_backward_options = {}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::Sigmoid.sigmoid(x)
    assert_equal(@dtype, y.data.class)
    y_expect = Chainer::Functions::Activation::Sigmoid.sigmoid(Chainer::Variable.new(@x))
    Chainer::Testing.assert_allclose(y_expect.data, y.data, @check_forward_options)
  end

  def test_forward
    check_forward(@x.dup)
  end

  def check_backward(x_data, y_grad, use_cudnn: "always")
    Chainer::check_backward(Chainer::Functions::Activation::Sigmoid.method(:sigmoid), x_data, y_grad, @check_backward_options)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad, use_cudnn: 'always')
    func = -> (x) do
      y = Chainer::Functions::Activation::Sigmoid.sigmoid(x)
      y * y
    end
    Chainer::check_double_backward(func, x_data, y_grad, x_grad_grad, @check_backward_options)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
