module Chainer
  module Functions
    module Array
      # Select elements stored in given indices.
      class SelectItem < FunctionNode
        # Select elements stored in given indices.
        #  This function returns $t.choose(x.T)$, that means
        #  $y[i] == x[i, t[i]]$ for all $i$.
        #
        #  @param [Chainer::Variable] x Variable storing arrays.
        #  @param [Chainer::Variable] t Variable storing index numbers.
        #  @return [Chainer::Variable] Variable that holds $t$-th element of $x$.
        def self.select_item(x, t)
          SelectItem.new.apply([x, t]).first
        end

        def forward(inputs)
          retain_inputs([1])
          x, t = inputs
          @in_shape = x.shape
          @in_dtype = x.class

          # TODO: x[six.moves.range(t.size), t]
          new_x = x.class.zeros(t.size)
          t.size.times.each do |i|
            new_x[i] = x[i, t[i]]
          end
          x = new_x

          [x]
        end

        def backward(indexes, gy)
          t = get_retained_inputs.first
          ret = []
          if indexes.include?(0)
            ggx = Assign.new(@in_shape, @in_dtype, t).apply(gy).first
            ret << ggx
          end
          if indexes.include?(1)
            ret << nil
          end
          ret
        end
      end

      class Assign < FunctionNode
        def initialize(shape, dtype, t)
          @shape = shape
          @dtype = dtype
          @t = t.data
        end

        def forward(inputs)
          gx = @dtype.zeros(*@shape)

          # TODO: gx[six.moves.range(self.t.size), self.t] = inputs[0]
          # binding.pry
          @t.size.times.each do |i|
            gx[i, @t[i]] = inputs[0][i]
          end

          [gx]
        end

        def backward(indexes, gy)
          SelectItem.new.apply([gy[0], @t])
        end
      end
    end
  end
end
