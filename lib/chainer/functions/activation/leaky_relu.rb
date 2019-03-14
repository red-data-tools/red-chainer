module Chainer
  module Functions
    module Activation
      # Leaky rectifier unit.
      class LeakyReLU < FunctionNode
        # Leaky Rectified Linear Unit function.
        #
        # This function is expressed as
        #
        # $$
        # f(x)=\\max(x, ax),
        # $$
        # 
        # where $a$ is a configurable slope value.
        # 
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @param [float] slope Slope value $a$.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example
        #   > x = Numo::SFloat[[-1, 0], [2, -3], [-2, 1]]
        #   > x
        #   => Numo::SFloat#shape=[3,2]
        #   [[-1, 0], 
        #    [2, -3], 
        #    [-2, 1]]
        #   > F = Chainer::Functions::Activation::LeakyReLU
        #   > F.leaky_relu(x, slope:0.2).data
        #   => Numo::SFloat#shape=[3,2]
        #   [[-0.2, 0], 
        #    [2, -0.6], 
        #    [-0.4, 1]]
        #
        def self.leaky_relu(x, slope: 0.2)
          self.new(slope: slope).apply([x])[0]
        end

        def initialize(slope:0.2)
          @slope = slope
        end

        def forward(inputs)
					x, = inputs
          y = x.dup
          y[x < 0] *= @slope
          if @slope >= 0
            retain_outputs([0])
          else
            retain_inputs([0])
          end
          [y]
        end

        def backward(indexes, grad_outputs)
          if @slope >= 0
            x = nil
            y = get_retained_outputs.first.data
          else
            x = get_retained_inputs.first.data
            y = nil
          end
          LeakyReLUGrad.new(x, y, @slope).apply(grad_outputs)
        end
      end

      class LeakyReLUGrad < FunctionNode
        def initialize(x, y, slope)
          @x = x
          @y = y
          @slope = slope
        end

        def forward(inputs)
          gy, = inputs
          gy = gy.dup
          if @slope >= 0
            gy[@y < 0] *= @slope
          else
            gy[@x < 0] *= @slope
          end
          [gy]
        end

        def backward(indexes, grad_outputs)
          LeakyReLUGrad.new(@x, @y, @slope).apply(grad_outputs)
        end
      end
    end
  end
end
