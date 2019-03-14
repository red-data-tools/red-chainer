# frozen_string_literal: true

require 'chainer/functions/math/sum'

class Chainer::Functions::Math::ExpTest < Test::Unit::TestCase
  data(:shape, [[3, 2], []],             keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @x = data[:dtype].new(data[:shape]).rand(-1, 1)
    @gy = data[:dtype].new(data[:shape]).rand(-1, 1)
    @ggy = data[:dtype].new(data[:shape]).rand(-1, 1)
  end

  def check_forward(x_data)
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Math::Exp.exp(x)

    assert_equal(x.data.class, y.data.class)

    Chainer::Testing.assert_allclose(xm::NMath.exp(@x), y.data, atol: 1e-7, rtol: 1e-7)
  end

  def test_forward
    check_forward(@x)
  end

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
end
