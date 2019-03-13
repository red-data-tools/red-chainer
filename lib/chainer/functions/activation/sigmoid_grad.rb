module Chainer
  module Functions
    module Activation
      # Logistic sigmoid gradient function.
      class SigmoidGrad < FunctionNode
        def initialize(inputs)
          @x, = inputs
        end

        def forward(inputs)
          retain_inputs([0, 1])
          y, gy = inputs
          one = 1
          [Utils::Array.force_array(gy * y * (one - y))]
        end

        def backward(indexes, grad_outputs)
          y, gy = get_retained_inputs
          g, = grad_outputs
          [g * gy * ( 1 -2 * y), g * y * (1 - y)]
        end
      end
    end
  end
end
