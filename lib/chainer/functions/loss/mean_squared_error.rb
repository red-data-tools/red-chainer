module Chainer
  module Functions
    module Loss
      # Mean squared error (a.k.a. Euclidean loss) function.
      class MeanSquaredError < FunctionNode
        # Mean squared error function.
        #
        # This function computes mean squared error between two variables. The mean
        # is taken over the minibatch. Note that the error is not scaled by 1/2.
        #
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x0 Input variable.
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x1 Input variable.
        # @return [Chainer::Variable] A variable holding an array representing the mean squared error of two inputs.
        #
        def self.mean_squared_error(x0, x1)
          self.new.apply([x0, x1]).first
        end

        def forward(inputs)
          retain_inputs([0, 1])
          diff = (inputs[0] - inputs[1]).flatten.dup
          [diff.class.cast(diff.dot(diff) / diff.size)]
        end

        def backward(indexes, gy)
          x0, x1 = get_retained_inputs
          diff = x0 - x1
          gy0 = Chainer::Functions::Array::BroadcastTo.broadcast_to(gy[0], diff.shape)
          gx0 = gy0 * diff * (2.0 / diff.size)

          ret = []
          if indexes.include?(0)
            ret << gx0
          end
          if indexes.include?(1)
            ret << -gx0
          end
          ret
        end
      end
    end
  end
end
