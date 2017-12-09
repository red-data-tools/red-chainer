module Chainer
  module Functions
    module Activation
      # Hyperbolic tangent function.
      class Tanh < Function
        # Elementwise hyperbolic tangent function.
        #
        # $$
        # f(x)=\\tanh(x).
        # $$
        #
        # @param [Chainer::Variable or Numo::DFloat] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example
        #   > x = Numo::DFloat.new(3).seq(-1, 2)
        #   => Numo::DFloat#shape=[3]
        #   [-1, 1, 3]
        #   > F = Chainer::Functions::Activation::Tanh
        #   > F.tanh(x).data
        #   => Numo::DFloat#shape=[3]
        #   [-0.761594, 0.761594, 0.995055]
        #
        def self.tanh(x)
          self.new.(x)
        end

        def forward_cpu(x)
          y = Utils::Array.force_array(Numo::NMath.tanh(x[0]))
          retain_inputs([])
          retain_outputs([0])
          return [y]
        end

        def backward_cpu(x, gy)
          y = @output_data[0]
          one = y.dtype.type(1)
          [Utils::Array.force_array(gy[0] * (one - y * y))]
        end
      end
    end
  end
end
