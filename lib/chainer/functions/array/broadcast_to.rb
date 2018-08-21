module Chainer
  module Functions
    module Array
      # Function that broadcasts an array to a new shape.
      class BroadcastTo < Function
        def initialize(shape)
            @shape = shape
        end

        def self.broadcast_to(xs, shape)
          self.new(shape).(xs)
        end

        def forward(xs)
          retain_inputs([])
          @input = xs.first
          [Chainer::Utils::Array.broadcast_to(@input, @shape)]
        end

        def backward(xs, grads)
          [backward_one(@input.shape, @input.class, grads.first)]
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

