module Chainer
  module Functions
    module Connection
      class LinearFunction < Chainer::Function
        def self.linear(x, w, b=nil)
          if b.nil?
            self.new.(x, w)
          else
            self.new.(x, w, b)
          end
        end

        def forward(inputs)
          x = as_mat(inputs[0])
          w = inputs[1]

          y = x.dot(w.transpose).cast_to(x.class)
          if inputs.size == 3
            b = inputs[2]
            y += b
          end
          return [y]
        end

        def backward(inputs, grad_outputs)
          x = as_mat(inputs[0])
          w = inputs[1]
          gy = grad_outputs[0]
          gx = gy.dot(w).cast_to(x.class).reshape(*inputs[0].shape)
          gw = gy.transpose.dot(x).cast_to(w.class)
          if inputs.size == 3
            gb = gy.sum(0)
            [gx, gw, gb]
          else
            [gx, gw]
          end
        end

        private

        def as_mat(x)
          return x if x.ndim == 2
          x.reshape(x.shape.first, true)
        end
      end
    end
  end
end
