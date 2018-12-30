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

          mask = xm::Bit.cast(x.flatten.to_a.map{|i| i == @ignore_label}).reshape(*x.shape)
          if mask.where.empty?
            return [Chainer::Utils::Array.take(w, x, axis: 0)]
          end
          x = x.dup
          x[mask.where] = 0
          y = Chainer::Utils::Array.take(w, x, axis: 0).dup

          ndindex = Chainer::Utils::Array.ndindex(y.shape.take(y.shape.size - 1))
          mask.where.each {|i| y[*ndindex[i], true] = 0 }

          [y]
        end

        def backward(inputs, grad_outputs)
          (x, w) = inputs
          gy = grad_outputs[0]
          gw = w.class.zeros(w.shape)
          ndindex = Chainer::Utils::Array.ndindex(gw.shape[0, gw.shape.size - 1])

          gy2 = gy.reshape(x.size, true)
          x.reshape(x.size).each_with_index do |ix, i|
            next if ix == @ignore_label
            gw[*ndindex[ix], true] = gw[*ndindex[ix], true] + gy2[i, true]
          end

          [nil, gw]
        end
      end
    end
  end
end
