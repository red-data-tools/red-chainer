module Chainer
  module Functions
    module Activation
      # Rectified Linear Unit.
      class Relu < FunctionNode
        # Rectified Linear Unit function.
        #
        # $$
        # f(x)=\\max(0, x).
        # $$
        #
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example
        #   > x = Numo::SFloat[[-1, 0], [2, -3], [-2, 1]]
        #   > (x < 0).any?
        #   => true
        #   > F = Chainer::Functions::Activation::Relu
        #   > y = F.relu(x)
        #   > (y.data < 0).any?
        #   => false
        #   > y.shape
        #   => [3, 2]
        #
        def self.relu(x)
          y, = self.new.apply([x])
          y
        end

        def forward(x)
          retain_outputs([0])
          [Utils::Array.force_array(x[0].class.maximum(x[0], 0))]
        end

        def backward(indexes, gy)
          y = get_retained_outputs.first
          ReLUGrad2.new(y).apply([gy[0]])
        end
      end
    end
  end
end
