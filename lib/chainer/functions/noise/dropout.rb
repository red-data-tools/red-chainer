module Chainer
  module Functions
    module Noise
      class Dropout < Chainer::FunctionNode
        attr_reader :mask
        # Drops elements of input variable randomly.
        #
        # This function drops input elements randomly with probability `ratio` and
        # scales the remaining elements by factor `1 / (1 - ratio)`.
        # In testing mode, it does nothing and just returns `x`.
        #
        # @param [Chainer::Variable] x Input variable.
        # @param [float] ratio Dropout ratio. The ``ratio`` must be `0.0 <= ratio < 1.0`.
        # @return [Chainer::Variable] Output variable.
        def self.dropout(x, ratio: 0.5)
          Chainer.configuration.train ? self.new(ratio).apply([x])[0] : x
        end

        def initialize(dropout_ratio)
          if dropout_ratio < 0 || dropout_ratio >= 1.0
            raise 'dropout_ratio must be in the range [0, 1)'
          end
          @dropout_ratio = dropout_ratio
        end

        def forward(x)
          unless self.instance_variable_defined?(:@mask)
            scale = x[0].class[*[1.0 / (1 - @dropout_ratio)]][0]
            flag = x[0].class.new(*x[0].shape).rand >= @dropout_ratio

            @mask = x[0].class.zeros(*x[0].shape)
            @mask[flag] = 1
            @mask *= scale
          end
          [x[0] * @mask]
        end

        def backward(x, gy)
          DropoutGrad.new(@mask).apply(gy)
        end
      end

      # Computes the gradient of the Dropout function.
      class DropoutGrad < Chainer::FunctionNode
        def initialize(mask)
          @mask = mask
        end

        def forward(inputs)
          y = inputs.first * @mask
          [y]
        end

        def backward(indexes, gy)
          DropoutGrad.new(@mask).apply(gy)
        end
      end
    end
  end
end

