# frozen_string_literal: true

require 'chainer/functions/array/transpose'

class Chainer::Functions::Array::TransposeTest < Test::Unit::TestCase
  data(:in_shape, [[4, 3, 2]], keep: true)
  data(:axes, [[0, 1], nil], keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @x = data[:dtype].new(*data[:in_shape]).rand
  end

  def test_forward(data)
    x = Chainer::Variable.new(@x)
    y = Chainer::Functions::Array::Transpose.transpose(x, axes: data[:axes])

    assert_equal(data[:dtype], y.dtype)
    assert_equal(@x.transpose(*data[:axes]), y.data)
  end

  def test_backward(data)
    x = Chainer::Variable.new(@x)

    y = Chainer::Functions::Array::Transpose.transpose(x, axes: data[:axes])
    y.grad = y.data
    y.backward

    Chainer::Testing.assert_allclose(x.data, x.grad, atol: 0, rtol: 0)
  end
end
