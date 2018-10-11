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
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example
        #   > x = Numo::SFloat.new(3).seq(-1, 2)
        #   => Numo::SFloat#shape=[3]
        #   [-1, 1, 3]
        #   > F = Chainer::Functions::Activation::Tanh
        #   > F.tanh(x).data
        #   => Numo::SFloat#shape=[3]
        #   [-0.761594, 0.761594, 0.995055]
        #
        def self.tanh(x)
          self.new.(x)
        end

        def forward_cpu(x)
          xm = Chainer.get_array_module(x[0])
          y = Utils::Array.force_array(xm::NMath.tanh(x[0]))
          retain_inputs([])
          retain_outputs([0])
          return [y]
        end

        def backward_cpu(x, gy)
          y = @output_data[0]
          one = y.class.cast(1)
          [Utils::Array.force_array(gy[0] * (one - y * y))]
        end
      end
    end
  end
end
