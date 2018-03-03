module Chainer
  module Functions
    module Loss
      # Mean squared error (a.k.a. Euclidean loss) function.
      class MeanSquaredError < Function
        # Mean squared error function.
        #
        # This function computes mean squared error between two variables. The mean
        # is taken over the minibatch. Note that the error is not scaled by 1/2.
        #
        # @param [Chainer::Variable or Numo::NArray] x0 Input variable.
        # @param [Chainer::Variable or Numo::NArray] x1 Input variable.
        # @return [Chainer::Variable] A variable holding an array representing the mean squared error of two inputs.
        #
        def self.mean_squared_error(x0, x1)
          self.new.(x0, x1)
        end

        def forward_cpu(inputs)
          x0, x1 = inputs
          @diff = x0 - x1
          diff = @diff.flatten.dup()
          [diff.class.cast(diff.dot(diff) / diff.size)]
        end

        def backward(inputs, gy)
          coeff = gy[0] * gy[0].class.cast(2.0 / @diff.size)
          gx0 = coeff * @diff
          [gx0, -(gx0)]
        end
      end
    end
  end
end
