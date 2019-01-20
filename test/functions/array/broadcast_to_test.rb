# frozen_string_literal: true

require 'chainer/functions/array/broadcast_to'

class Chainer::Functions::Array::BroadcastToTest < Test::Unit::TestCase
  data(:shape, [
    { in_shape: [3, 1, 5], out_shape: [3, 2, 5] },
    { in_shape: [5,],      out_shape: [3, 2, 5] },
    { in_shape: [3, 2, 5], out_shape: [3, 2, 5] }
  ], keep: true)
  data(:dtype, [ xm::SFloat, xm::DFloat ], keep: true)

  def test_forward(data)
    in_data = data[:dtype].new(data[:shape][:in_shape]).rand
    x = Chainer::Variable.new(in_data)
    bx = Chainer::Functions::Array::BroadcastTo.broadcast_to(x, data[:shape][:out_shape])

    assert_equal(bx.data.shape, data[:shape][:out_shape])
  end

  def test_backward(data)
    in_data = data[:dtype].new(data[:shape][:in_shape]).rand
    grads = data[:dtype].new(data[:shape][:out_shape]).rand
    check_backward_options = {}
    if data[:dtype] == xm::SFloat
      check_backward_options = { eps: 2 ** -5, atol: 1e-3, rtol: 1e-2 }
    end

    func = Chainer::Functions::Array::BroadcastTo.new(data[:shape][:out_shape])
    Chainer::check_backward(func, in_data, grads, **check_backward_options)
  end
end
