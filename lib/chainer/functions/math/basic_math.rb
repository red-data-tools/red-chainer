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
 
      class Sub < ::Chainer::Function
        def forward(x)
          retain_inputs([])
          [Utils::Array.force_array(x[0] - x[1])]
        end

        def backward(x, gy)
          [gy[0], Utils::Array.force_array(-gy[0])]
        end
      end

      class Mul < ::Chainer::Function
        def forward(x)
          [Utils::Array.force_array(x[0] * x[1])]
        end

        def backward(x, gy)
          [Utils::Array.force_array(gy[0] * x[1]), Utils::Array.force_array(gy[0] * x[0])]
        end
      end

      class MulConstant < ::Chainer::Function
        def initialize(value)
          @value = value
        end

        def forward(x)
          [Utils::Array.force_array(@value * x[0])]
        end

        def backward(x, gy)
          [Utils::Array.force_array(@value * gy[0])]
        end
      end
      
      class Div < ::Chainer::Function
        def forward(x)
          [Utils::Array.force_array(x[0] / x[1])]
        end

        def backward(x, gy)
          gx0 = Utils::Array.force_array(gy[0] / x[1])
          [gx0, Utils::Array.force_array(-1 * gx0 * x[0] / x[1])]
        end
      end
      
      class PowVarVar < ::Chainer::Function
        def forward(x)
          @y = Utils::Array.force_array(x[0] ** x[1])
          [@y]
        end

        def backward(x, gy)
          one = x[1].class.ones[0]
          gx0 = Utils::Array.force_array(x[1] * (x[0] ** (x[1] - one)) * gy[0])
          gx1 = Utils::Array.force_array(Numo::NMath.log(x[0]) * @y * gy[0])
          [gx0, gx1]
        end
      end

      class PowVarConst < ::Chainer::Function
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
