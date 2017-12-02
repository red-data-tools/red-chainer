module Chainer
  module Functions
    module Activation
      # Logistic sigmoid function.
      class Sigmoid < Function
        # Element-wise sigmoid logistic function.
        #
        # $$
        # f(x)=(1 + \\exp(-x))^ { -1 }.
        # $$
        #
        # @param [Chainer::Variable or Numo::DFloat] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example  It maps the input values into the range of $`[0, 1]`$.
        #   > x = Numo::DFloat.new(3).seq(-2, 2)
        #   => Numo::DFloat#shape=[3]
        #   [-2, 0, 2]
        #   > F = Chainer::Functions::Activation::Sigmoid
        #   > F.sigmoid(x).data
        #   => Numo::DFloat#shape=[3]
        #   [0.119203, 0.5, 0.880797]
        #
        def self.sigmoid(x)
          self.new.(x)
        end

        def forward_cpu(x)
          half = 0.5
          y = Utils::Array.force_array((Numo::NMath.tanh(x[0] * half) * half)+ half)
          retain_inputs([])
          retain_outputs([0])
          return [y]
        end

        def backward_cpu(x, gy)
          one = 1
          y = @output_data[0]
          [Utils::Array.force_array((gy[0] * y) * (one - y))]
        end
      end
    end
  end
end
