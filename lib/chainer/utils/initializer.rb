module Chainer
  module Utils
    module Initializer
      def self.get_fans(shape)
        raise 'shape must be of length >= 2: shape={}' if shape.size < 2
        slice_arr = shape.slice(2, shape.size)
        receptive_field_size = slice_arr.empty? ? 1 : Numo::Int32[slice_arr].prod
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
        [fan_in, fan_out]
      end
    end
  end
end
