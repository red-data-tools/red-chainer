module Chainer
  module Functions
    module Activation
      # Computes the gradient of the ReLU function.
      #
      # This function takes 2 variables b and c, and
      # computes f(b, c) = sign(b) * c with backpropagation
      # where operations are dones in elementwise manner
      # and sign(x) = 1 when x > 0 is positive and 0 otherwise.
      # As the gradient of f with respect to b is 0,
      # we do not backpropagate errors toward b for computational efficiency.<Paste>
      class ReLUGrad2 < FunctionNode
        def initialize(b)
          @b = b.data
        end

        def forward(inputs)
          y = inputs[0] * (@b > 0)
          [Utils::Array.force_array(y, y.class)]
        end

        def backward(indexes, gy)
          [gy[0] * heaviside(@b)]
        end

        private

        def heaviside(x)
          (x > 0).cast_to(x.class)
        end
      end
    end
  end
end
