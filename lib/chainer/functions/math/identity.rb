module Chainer
  module Functions
    module Math
      # Identity function.
      class Identity < Chainer::Function
        def check_type_forward(in_types)
          # pass
        end

        def forward(xs)
          retain_inputs([])
          return xs
        end

        def backward(xs, gys)
          return gys
        end

        # Just returns input variables.
        def self.identity(*inputs)
          self.new.(*inputs)
        end
      end
    end
  end
end
