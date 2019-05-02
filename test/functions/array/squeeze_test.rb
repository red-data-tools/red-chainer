# frozen_string_literal: true

require 'chainer/functions/array/squeeze'

class Chainer::Functions::Array::SqueezeTest < Test::Unit::TestCase
  data(:test_case, [
    { axis: nil, in_shape: [1, 3, 1, 3], out_shape: [3, 3] },
    { axis: 0, in_shape: [1, 3, 1, 3], out_shape: [3, 1, 3] },
    { axis: [2], in_shape: [1, 3, 1, 3], out_shape: [1, 3, 3] }
  ], keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def test_forward
    in_data = data[:dtype].new(data[:test_case][:in_shape]).seq
    x = Chainer::Variable.new(in_data)
    y = Chainer::Functions::Array::Squeeze.squeeze(x, axis: data[:test_case][:axis])

    assert_equal(y.data.shape, data[:test_case][:out_shape])
    assert_equal(y.dtype, data[:dtype])
  end

  def test_backward
    in_data = data[:dtype].new(data[:test_case][:in_shape]).seq
    x = Chainer::Variable.new(in_data)
    y = Chainer::Functions::Array::Squeeze.squeeze(x, axis: data[:test_case][:axis])
    y.grad = y.data
    y.backward

    Chainer::Testing.assert_allclose(x.data, x.grad, atol: 0, rtol: 0)
  end
end
