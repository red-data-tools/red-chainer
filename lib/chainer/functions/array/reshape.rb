module Chainer
  module Functions
    module Array
      # Reshapes an input array without copy.
      class Reshape < FunctionNode
        def initialize(shape)
          @shape = shape
        end

        def self.reshape(x, shape)
          if x.shape == shape
            return x if x.is_a?(Chainer::Variable)
            return Chainer::Variable.new(x, requires_grad: false)
          end
          y = self.new(shape).apply([x]).first
          y
        end

        def forward(inputs)
          x = inputs.first
          [x.reshape(*@shape)]
        end

        def backward(indexes, grad_outputs)
          gx = grad_outputs.first
          [self.reshape(gx, @inputs.first.shape)]
        end
      end
    end
  end
end
