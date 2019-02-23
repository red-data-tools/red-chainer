module Chainer
  module Functions
    module Array
      class Cast < FunctionNode
        # Cast an input variable to a given type.
        #
        # @param x [Chainer::Variable or Numo::Narray] x : Input variable to be casted.
        # @param type [Numo::Narray class] type : data class to cast
        # @return [Chainer::Variable] Variable holding a casted array.
        #
        # example
        # > x = Numo::UInt8.new(3, 5).seq
        # > x.class
        # # => Numo::UInt8
        # > y = Chainer::Functions::Array::Cast.cast(x, Numo::DFloat)
        # > y.dtype
        # # => Numo::DFloat
        def self.cast(x, type)
          self.new(type).apply([x]).first
        end

        def initialize(type)
            @type = type
        end

        def forward(x)
          @in_type = x.first.class
          [x.first.cast_to(@type)]
        end

        def backward(indexes, g)
          [Cast.cast(g.first, @in_type)]
        end
      end
    end
  end
end

