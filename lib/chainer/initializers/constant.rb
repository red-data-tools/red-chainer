module Chainer
  module Initializers
    class Constant < ::Chainer::Initializer
      def initialize(fill_value, dtype: nil)
        @fill_value = fill_value
        super(dtype: dtype)
      end

      def call(array)
        Numo::NArray.cast(array)
      end
    end
  end
end
