module Chainer
  module Functions
    module Array
      # Reshapes an input array without copy.
      class Reshape < FunctionNode
        def initialize(shape)
          @shape = shape
        end

        def self.reshape(x, shape)
          return Chainer::Variable.as_variable(x) if x.shape == shape
          return self.new(shape).apply([x]).first
        end

        def forward(inputs)
          x = inputs.first
          [x.reshape(*@shape)]
        end

        def backward(indexes, grad_outputs)
          gx = grad_outputs.first
          [Reshape.reshape(gx, @inputs.first.shape)]
        end
      end
    end
  end
end
