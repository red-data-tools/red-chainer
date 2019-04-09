# frozen_string_literal: true

require 'chainer/functions/math/sum'

class Chainer::Functions::Math::SumTest < Test::Unit::TestCase
  data(:axis, [nil, 0, 1, 2, -1, [0, 1], [1, 0], [0, -1], [-2, 0]], keep: true)
  data(:keepdims, [true, false], keep: true)
  data(:dtype, [ xm::SFloat, xm::DFloat ], keep: true)

  def test_forward(data)
    x_data = data[:dtype].new([3, 2, 4]).rand
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Math::Sum.sum(x, axis: data[:axis], keepdims: data[:keepdims])
    assert_equal(y.data.class, data[:dtype])

    y_expect = x_data.sum(axis: data[:axis], keepdims: data[:keepdims])
    Chainer::Testing.assert_allclose(y_expect, y.data, atol: 0, rtol: 0)
  end

  def test_backward(data)
    x = data[:dtype].new([3, 2, 4]).rand

    g = x.sum(axis: data[:axis], keepdims: data[:keepdims])
    g_shape = g.is_a?(xm::NArray) ? g.shape : []
    gy = data[:dtype].new(g_shape).rand

    func = lambda{ |x| Chainer::Functions::Math::Sum.sum(x, axis: data[:axis], keepdims: data[:keepdims]) }
    Chainer::check_backward(func, x, gy, atol: 1e-4, dtype: xm::DFloat)
  end
end
