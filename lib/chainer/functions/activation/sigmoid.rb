module Chainer
  module Functions
    module Activation
      # Logistic sigmoid function.
      class Sigmoid < FunctionNode
        # Element-wise sigmoid logistic function.
        #
        # $$
        # f(x)=(1 + \\exp(-x))^ { -1 }.
        # $$
        #
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example  It maps the input values into the range of $`[0, 1]`$.
        #   > x = Numo::SFloat.new(3).seq(-2, 2)
        #   => Numo::SFloat#shape=[3]
        #   [-2, 0, 2]
        #   > F = Chainer::Functions::Activation::Sigmoid
        #   > F.sigmoid(x).data
        #   => Numo::SFloat#shape=[3]
        #   [0.119203, 0.5, 0.880797]
        #
        def self.sigmoid(x)
          self.new.apply([x]).first
        end

        def forward(inputs)
          x, = inputs
          half = 0.5
          xm = Chainer.get_array_module(x)
          y = Utils::Array.force_array((xm::NMath.tanh(x * half) * half)+ half)
          retain_outputs([0])
          [y]
        end

        def backward(indexes, grad_outputs)
          x = nil
          y = get_retained_outputs.first
          gy, = grad_outputs
          Chainer::Functions::Activation::SigmoidGrad.new([x]).apply([y, gy])
        end
      end
    end
  end
end
