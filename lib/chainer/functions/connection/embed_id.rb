module Chainer
  module Functions
    module Connection
      class EmbedIDFunction < Chainer::Function
        def initialize(ignore_label: nil)
          @ignore_label = ignore_label
        end

        def self.embed_id(x, w, ignore_label: nil)
          self.new(ignore_label: ignore_label).(x, w)
        end

        def forward(inputs)
          xm = Chainer.get_array_module(*inputs)
          (x, w) = inputs

          unless @ignore_label
            return [Chainer::Utils::Array.take(w, x, axis: 0)]
          end

          valid_x = x.ne(@ignore_label)
          if valid_x.count == x.size
            return [Chainer::Utils::Array.take(w, x, axis: 0)]
          end
          x *= valid_x
          y = Chainer::Utils::Array.take(w, x, axis: 0).dup

          y = y.reshape(y.shape.take(y.shape.size - 1).reduce(&:*), true)
          valid_x.where2.last.each {|i| y[i, true] = y.class.zeros(y.shape.last) }

          [y.reshape(*x.shape, true)]
        end

        def backward(inputs, grad_outputs)
          (x, w) = inputs
          gy = grad_outputs[0].reshape(x.size, true)
          gw = w.class.zeros(w.shape).reshape(w.shape.take(w.shape.size - 1).reduce(&:*), true)

          x.reshape(x.size).each_with_index do |ix, i|
            next if ix == @ignore_label
            gw[ix, true] = gw[ix, true] + gy[i, true]
          end

          [nil, gw.reshape(*w.shape)]
        end
      end
    end
  end
end
