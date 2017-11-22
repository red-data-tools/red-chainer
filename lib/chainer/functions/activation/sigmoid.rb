module Chainer
  module Functions
    module Activation
      # Logistic sigmoid function.
      class Sigmoid < Function
        # Element-wise sigmoid logistic function.
        #
        #  (ToDo. This is Sphinx format, but I would like to convert it to YARD(and MathJax).) 
        #
        #    .. math:: f(x)=(1 + \\exp(-x))^{-1}.
        #
        #   Args:
        #       x (:class:`~Chainer::Variable.new` or :class:`Numo::DFloat`):
        #           Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        #
        #   Returns:
        #       ~Chainer::Variable: Output variable. A
        #       :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        #
        #   .. admonition:: Example
        #
        #       It maps the input values into the range of :math:`[0, 1]`.
        #
        #       > x = Numo::DFloat.new(3).seq(-2, 2)
        #       => Numo::DFloat#shape=[3]
        #       [-2, 0, 2]
        #       > F = Chainer::Functions::Activation::Sigmoid
        #       > F.sigmoid(x).data
        #       => Numo::DFloat#shape=[3]
        #       [0.119203, 0.5, 0.880797]
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
