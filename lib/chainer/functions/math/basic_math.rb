module Chainer
  module Functions
    module Math
      class Neg < ::Chainer::FunctionNode
        def label
          '__neg__'
        end

        def forward(x)
          [Utils::Array.force_array(-x[0])]
        end

        def backward(indexes, gy)
          [-gy[0]]
        end
      end

      class Add < ::Chainer::FunctionNode
        def forward(x)
          [Utils::Array.force_array(x[0] + x[1])]
        end

        def backward(indexes, gy)
          [gy[0], gy[0]]
        end
      end

      class AddConstant < ::Chainer::FunctionNode
        def initialize(value)
          @value = value
        end

        def forward(x)
          [Utils::Array.force_array(x[0] + @value)]
        end

        def backward(indexes, gy)
          [gy[0]]
        end
      end

      class Sub < ::Chainer::FunctionNode
        def label
          '_ - _'
        end

        def forward(x)
          [Utils::Array.force_array(x[0] - x[1])]
        end

        def backward(indexes, gy)
          [gy[0], -gy[0]]
        end
      end

      class Mul < ::Chainer::FunctionNode
        def forward(x)
          retain_inputs([0, 1])
          [Utils::Array.force_array(x[0] * x[1])]
        end

        def backward(indexes, gy)
          xs = get_retained_inputs
          indexes.map { |i| gy[0] * xs[1 - i] }
        end
      end

      class MulConstant < ::Chainer::FunctionNode
        def initialize(value)
          @value = value
        end

        def forward(x)
          [Utils::Array.force_array(@value * x[0])]
        end

        def backward(indexes, gy)
          [@value * gy[0]]
        end
      end

      class Div < ::Chainer::FunctionNode
        def forward(x)
          [Utils::Array.force_array(x[0] / x[1])]
        end

        def backward(indexes, gy)
          gx0 = Utils::Array.force_array(gy[0] / x[1])
          [gx0, Utils::Array.force_array(-1 * gx0 * x[0] / x[1])]
        end
      end

      class PowVarVar < ::Chainer::FunctionNode
        def forward(x)
          @y = Utils::Array.force_array(x[0] ** x[1])
          [@y]
        end

        def backward(x, gy)
          one = x[1].class.ones[0]
          gx0 = Utils::Array.force_array(x[1] * (x[0] ** (x[1] - one)) * gy[0])
          xm = Chainer.get_array_module(x[0])
          gx1 = Utils::Array.force_array(xm::NMath.log(x[0]) * @y * gy[0])
          [gx0, gx1]
        end
      end

      class PowVarConst < ::Chainer::FunctionNode
        def initialize(value)
          @value = value
        end

        def forward(x)
          [Utils::Array.force_array(x[0] ** @value)]
        end

        def backward(x, gy)
          val_1 = @value - 1
          gx = @value * (x[0] ** val_1) * gy[0]
          [Utils::Array.force_array(gx)]
        end
      end
    end
  end
end
