module Chainer
  module Functions
    module Math
      # Identity function.
      class Identity < Chainer::FunctionNode
        def check_type_forward(in_types)
          # pass
        end

        def forward(xs)
          retain_inputs([])
          return xs
        end

        def backward(indexes, gys)
          return gys
        end

        # Just returns input variables.
        def self.identity(*inputs)
          ret = self.new.apply(*inputs)
          ret.size == 1 ? ret[0] : ret
        end
      end
    end
  end
end
