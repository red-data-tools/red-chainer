module Chainer
  module Initializers
    class Constant < ::Chainer::Initializer
      def initialize(fill_value, dtype: nil)
        @fill_value = fill_value
        super(dtype: dtype)
      end

      def call(array)
        if @dtype
          raise ArgumentError unless array.class == @dtype
        end
        array.store(@fill_value)
        array
      end
    end
  end
end
