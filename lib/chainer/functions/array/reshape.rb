module Chainer
  module Functions
    module Array
      # Reshapes an input array without copy.
      class Reshape < Function
        def initialize(shape)
            @shape = shape
        end

        def self.reshape(xs, shape)
          self.new(shape).(xs)
        end

        def forward(xs)
          retain_inputs([])

          input = xs.first
          @input_shape = input.shape
          [input.reshape(*@shape)]
        end

        def backward(xs, grads)
          [grads.first.reshape(*@input_shape)]
        end
      end
    end
  end
end
