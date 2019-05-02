module Chainer
  module Functions
    module Array
      # Function that broadcasts an array to a new shape.
      class BroadcastTo < FunctionNode
        def initialize(shape)
            @shape = shape
        end

        def self.broadcast_to(x, shape)
          return Chainer::Variable.as_variable(x) if x.shape == shape
          self.new(shape).apply([x]).first
        end

        def forward(inputs)
          x = inputs.first
          [Chainer::Utils::Array.broadcast_to(x, @shape)]
        end

        def backward(indexes, grad_outputs)
          gx = grad_outputs.first
          shape = @inputs.first.shape
          ndim = shape.size
          lead = gx.ndim - ndim
          lead_axis = lead.times.to_a
          axis = shape.each_with_object([]).with_index do |(sx, res), i|
            next unless sx == 1
            res << i + lead
          end
          gx = Chainer::Functions::Math::Sum.sum(gx, axis: lead_axis + axis, keepdims: true)
          return [Chainer::Functions::Array::Squeeze.squeeze(gx, axis: lead_axis)] if lead > 0
          [gx]
        end

        private

        def backward_one(shape, dtype, g)
          return dtype.zeros(shape) unless g

          ndim = shape.size
          if g.ndim != ndim
            g = g.sum(axis: 0...(g.ndim - ndim))
          end

          axis = shape.each_with_index.select{|sx, i| sx == 1 }.map{|sx, i| i }
          if axis.size > 0
            g.sum(keepdims: true, axis: axis)
          else
            g
          end
        end
      end
    end
  end
end

