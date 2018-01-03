module Chainer
  module Functions
    module Activation
      # Leaky rectifier unit.
      class LeakyReLU < Function
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
        # @param [Chainer::Variable or Numo::DFloat] x Input variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @param [float] slope Slope value $a$.
        # @return [Chainer::Variable] Output variable. A $(s_1, s_2, ..., s_N)$-shaped float array.
        # @example
        #   > x = Numo::DFloat[[-1, 0], [2, -3], [-2, 1]]
        #   > x
        #   => Numo::DFloat#shape=[3,2]
        #   [[-1, 0], 
        #    [2, -3], 
        #    [-2, 1]]
        #   > F = Chainer::Functions::Activation::LeakyReLU
        #   > F.leaky_relu(x, slope:0.2).data
        #   => Numo::DFloat#shape=[3,2]
        #   [[-0.2, 0], 
        #    [2, -0.6], 
        #    [-0.4, 1]]
        #
        def self.leaky_relu(x, slope: 0.2)
          self.new(slope: slope).(x)
        end

        def initialize(slope:0.2)
          @slope = slope
        end

        def forward_cpu(x)
          y = x[0].dup()
          y[x[0] < 0] *= @slope
          if @slope >= 0
            retain_inputs([])
            retain_outputs([0])
          end
          [y]
        end

        def backward_cpu(x, gy)
          gx = gy[0].dup()
          if @slope >= 0
            y = @output_data
            gx[y[0] < 0] *= @slope
          else
            gx[x[0] < 0] *= @slope
          end
          [gx]
        end
      end
    end
  end
end
