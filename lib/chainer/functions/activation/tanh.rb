module Chainer
  module Functions
    module Activation
      # Hyperbolic tangent function.
      class Tanh < FunctionNode
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
          self.new.apply([x]).first
        end

        def forward(x)
          xm = Chainer.get_array_module(x[0])
          y = Utils::Array.force_array(xm::NMath.tanh(x[0]))
          retain_outputs([0])
          @use_cudnn = false
          [y]
        end

        def backward(indexes, grad_outputs)
          if @use_cudnn
            x = get_retained_inputs.first.data
          else
            x = nil
          end

          y = get_retained_outputs.first
          gy = grad_outputs.first
          TanhGrad.new(x).apply([y, gy])
        end
      end

      class TanhGrad < FunctionNode
        def initialize(x)
          super()

          # The original input `x` is only required for cuDNN.
          # If it is None, this class does not use cuDNN.
          # Note that x must be c-contiguous and it is checked
          # in Tanh.forward_gpu.
          @x = x
        end

        def forward(inputs)
          retain_inputs([0, 1])
          y, gy = inputs

          one = y.class.new.fill(1)
          [Utils::Array.force_array(gy * (one - y * y))]
        end

        def backward(indexes, grad_outputs)
          y, gy = get_retained_inputs
          g = grad_outputs[0]

          y_mul_g = y * g
          grad_y = -2 * gy * y_mul_g
          ggy = g - y * y_mul_g
          [grad_y, ggy]
        end
      end
    end
  end
end
