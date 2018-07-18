# frozen_string_literal: true

require 'chainer/functions/array/broadcast_to'

class Chainer::Functions::Array::BroadcastToTest < Test::Unit::TestCase
  shapes = [
    {
      in_shape: [3, 1, 5],
      out_shape: [3, 2, 5]
    },
    {
      in_shape: [5,],
      out_shape: [3, 2, 5]
    },
    {
      in_shape: [3, 2, 5],
      out_shape: [3, 2, 5]
    }
  ]

  dtypes = [ Numo::SFloat, Numo::DFloat ]

  data =  shapes.map.with_index {|shape, i|
            dtypes.map do |dtype|
              ["shape#{i}:#{dtype}", shape.merge(dtype: dtype)]
            end
          }.flatten(1).to_h

  data(data)
  def test_forward_cpu(data)
    in_data = data[:dtype].new(data[:in_shape]).rand
    x = Chainer::Variable.new(in_data)
    bx = Chainer::Functions::Array::BroadcastTo.broadcast_to(x, data[:out_shape])

    assert_equal(bx.data.shape, data[:out_shape])
  end

  data(data)
  def test_backward_cpu(data)
    in_data = data[:dtype].new(data[:in_shape]).rand
    grads = data[:dtype].new(data[:out_shape]).rand
    check_backward_options = {}
    if data[:dtype] == Numo::SFloat
        check_backward_options = { eps: 2 ** -5, atol: 1e-3, rtol: 1e-2 }
    end

    func = Chainer::Functions::Array::BroadcastTo.new(data[:out_shape])
    Chainer::check_backward(func, in_data, grads, **check_backward_options)
  end
end
