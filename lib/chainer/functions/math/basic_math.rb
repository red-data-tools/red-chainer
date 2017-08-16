module Chainer
  module Functions
    module Math
      class Add < ::Chainer::Function
        def forward(x)
          retain_inputs([])
          [Utils::Array.force_array(x[0] + x[1])]
        end

        def backward(x, gy)
          [gy[0], gy[0]]
        end
      end

      class AddConstant < ::Chainer::Function
        def initialize(value)
          @value = value
        end

        def forward(x)
          retain_inputs([])
          [Utils::Array.force_array(x[0] + @value)]
        end

        def backward(x, gy)
          [gy[0]]
        end
      end
    end
  end
end
