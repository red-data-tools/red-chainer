module Chainer
  module Functions
    module Math
      class Exp < Chainer::FunctionNode
        # Elementwise exponential function.
        def self.exp(x)
          self.new.apply([x]).first
        end

        def label
          'exp'
        end

        def forward(x)
          retain_inputs([])
          retain_outputs([0])
          [Utils::Array.force_array(xm::NMath.exp(x.first))]
        end

        def backward(indexes, gy)
        	y = get_retained_outputs.first
          [y * gy.first]
        end
      end
    end
  end
end
