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

        def forward_cpu(inputs)
          (x, w) = inputs

          unless @ignore_label
            return [Chainer::Utils::Array.take(w, x.to_a, axis: 0)]
          end

          mask = Numo::Bit.cast(x.flatten.to_a.map{|i| i == @ignore_label}).reshape(*x.shape)
          if mask.where.empty?
            return [Chainer::Utils::Array.take(w, x.to_a, axis: 0)]
          end
          x = x.dup
          x[mask.where] = 0
          y = Chainer::Utils::Array.take(w, x.to_a, axis: 0).dup

          ndindex = Chainer::Utils::Array.ndindex(y.shape.take(y.shape.size - 1))
          mask.where.each {|i| y[*ndindex[i], true] = 0 }

          [y]
        end

        def backward_cpu(inputs, grad_outputs)
          (x, w) = inputs
          gy = grad_outputs[0]
          gw = w.class.zeros(w.shape)
          ndindex = Chainer::Utils::Array.ndindex(gw.shape[0, gw.shape.size - 1])

          x.to_a.flatten.zip(gy.reshape(x.size, true).to_a).each do |ix, igy|
            next if ix == @ignore_label
            gw[*ndindex[ix], true] = gw[*ndindex[ix], true] + igy
          end

          [nil, gw]
        end
      end
    end
  end
end
