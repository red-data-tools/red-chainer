module Chainer
  module Functions
    module Connection
      class LinearFunction < Chainer::FunctionNode
        def self.linear(x, w, b=nil)
          if x.ndim > 2
            x = x.reshape(x.shape.first, -1)
          end

          if b.nil?
            args = x, w
          else
            args = x, w, b
          end

          self.new.apply(args).first
        end

        def forward(inputs)
          x = inputs[0]
          w = inputs[1]

          if x.shape[0] == 0
            y = x.class.new(0, w.shape[0])
            return [y]
          end

          y = x.dot(w.transpose).cast_to(x.class)

          if inputs.size == 3
            b = inputs[2]
            y += b
          end

          retain_inputs([0, 1])
          return [y]
        end

        def backward(indexes, grad_outputs)
          x, w = get_retained_inputs
          gy = grad_outputs.first

          ret = []
          if indexes.include?(0)
            gx = LinearFunction.linear(gy, w.transpose)
            ret << Chainer::Functions::Array::Cast.cast(gx, x.dtype)
          end
          if indexes.include?(1)
            gw = LinearFunction.linear(gy.transpose, x.transpose)
            ret << Chainer::Functions::Array::Cast.cast(gw, w.dtype)
          end
          if indexes.include?(2)
            gb = Chainer::Functions::Math::Sum.sum(gy, axis: 0)
            ret << gb
          end
          ret
        end
      end
    end
  end
end
