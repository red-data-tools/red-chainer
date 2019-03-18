# frozen_string_literal: true

class Chainer::Functions::Noise::DropoutTest < Test::Unit::TestCase
  def setup
    @dtype = Numo::SFloat
    @ratio = 0.3

    @x = @dtype.new([2, 3]).rand(-1, 1)
    @gy = @dtype.new([2, 3]).rand(-1, 1)
    @ggx = @dtype.new([2, 3]).rand(-1, 1)

    @check_backward_options = { dtype: Numo::DFloat }
    @check_double_backward_options = { dtype: Numo::DFloat }
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
    f = -> (x) do
      Chainer::Functions::Noise::Dropout.dropout(x, ratio: @ratio)
    end
    Chainer::check_backward(f, x_data, y_grad, **@check_backward_options)
  end

  def test_backward
    check_backward(@x, @gy)
  end




=begin
  def check_backward(x_data, y_grad)
    func = -> (x) do
      Chainer::Functions::Math::Exp.exp(x)
    end

    Chainer::check_backward(func, x_data, y_grad, atol: 1e-4, rtol: 1e-3, dtype: xm::DFloat)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad)
    func = -> (x) do
      Chainer::Functions::Math::Exp.exp(x)
    end

    Chainer::check_double_backward(func, x_data, y_grad, x_grad_grad, atol: 1e-4, rtol: 1e-3, dtype: xm::DFloat)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggy)
  end

  def test_label
    label = Chainer::Functions::Math::Exp.new.label
    assert_equal('exp', label)
  end
=end
end
