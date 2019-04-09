# frozen_string_literal: true

class Chainer::Functions::Noise::DropoutTest < Test::Unit::TestCase
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)
  data(:ratio, [0.0, 0.3, 0.5], keep: true)

  def setup
    @dtype = data[:dtype]
    @ratio = data[:ratio]

    @x = @dtype.new([2, 3]).rand(-1, 1)
    @gy = @dtype.new([2, 3]).rand(-1, 1)
    @ggx = @dtype.new([2, 3]).rand(-1, 1)

    @check_backward_options = { dtype: xm::DFloat }
    @check_double_backward_options = { dtype: xm::DFloat }
  end

  def check_forward(x_data)
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Noise::Dropout.dropout(x, ratio: @ratio)
    if @ratio == 0.0
      y_expect = x_data
    else
      y_expect = x_data * y.creator_node.mask
    end
    Chainer::Testing.assert_allclose(y_expect, y.data)
  end

  def test_forward
    check_forward(@x)
  end

  def check_backward(x_data, y_grad)
    dropout = Chainer::Functions::Noise::Dropout.new(@ratio)
    f = -> (x) do
      dropout.apply([x]).first
    end
    Chainer::check_backward(f, x_data, y_grad, **@check_backward_options)
  end

  def test_backward
    check_backward(@x, @gy)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad)
    dropout = Chainer::Functions::Noise::Dropout.new(@ratio)
    f = -> (x) do
      x, = dropout.apply([x])
      x * x
    end
    Chainer::check_double_backward(f, x_data, y_grad, x_grad_grad, **@check_double_backward_options)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
