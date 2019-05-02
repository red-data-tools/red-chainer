# frozen_string_literal: true

require 'chainer/functions/connection/linear'

class Chainer::Functions::Connection::LinearTest < Test::Unit::TestCase
  data(:dtype, [ xm::SFloat, xm::DFloat ], keep: true)

  def setup
    @w = data[:dtype].new(2, 3).rand
    @b = data[:dtype].new(2).rand

    @x = data[:dtype].new(4, 3).rand
    @gy = data[:dtype].new(4, 2).rand
    @ggx = data[:dtype].new(*@x.shape).rand
    @ggw = data[:dtype].new(*@w.shape).rand
    @ggb = data[:dtype].new(*@b.shape).rand

    @y = @x.dot(@w.transpose) + @b
  end

  def test_forward(data)
    x = Chainer::Variable.new(@x)
    w = Chainer::Variable.new(@w)
    b = Chainer::Variable.new(@b)
    y = Chainer::Functions::Connection::LinearFunction.linear(x, w, b)

    assert_equal(data[:dtype], y.data.class)

    y_expect = @x.dot(@w.transpose) + @b
    Chainer::Testing.assert_allclose(y_expect, y.data)
  end

  def test_backward(data)
    args = [@x, @w, @b]
    func = -> (x, w, b) { Chainer::Functions::Connection::LinearFunction.linear(x, w, b) }
    Chainer::check_backward(func, args, @gy)
  end

  def test_backward_nobias(data)
    args = [@x, @w]
    func = -> (x, w) { Chainer::Functions::Connection::LinearFunction.linear(x, w) }
    Chainer::check_backward(func, args, @gy)
  end

  def test_double_backward(data)
    args = [@x, @w, @b]
    grad_grads = [@ggx, @ggw, @ggb]

    func = -> (x, w, b) do
      y = Chainer::Functions::Connection::LinearFunction.linear(x, w, b)
      y * y
    end

    Chainer::check_double_backward(func, args, [@gy], grad_grads)
  end
end
