module Chainer
  module Initializers
    class Uniform < ::Chainer::Initializer
      def initialize(scale: 0.05, dtype: nil)
        @scale = scale
        super(dtype: dtype)
      end

      def call(array)
        raise ArgumentError.new("dtypes are missmatched. #{dtype} != #{array.class}") if dtype && dtype != array.class
        array.class.new(array.shape).rand(-@scale, @scale)
      end
    end
  end
end
