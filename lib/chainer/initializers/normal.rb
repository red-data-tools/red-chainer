module Chainer
  module Initializers
    class Normal < ::Chainer::Initializer
      def initialize(scale: 0.05, dtype: nil)
        @scale = scale
        super(dtype: dtype)
      end

      def call(array)
        args = { loc: 0.0, scale: @scale, size: array.shape}
        array.class.new(array.shape).rand_norm(0.0, @scale)
      end
    end

    class HeNormal < ::Chainer::Initializer
      def initialize(scale: 1.0, dtype: nil)
        @scale = scale
        super(dtype: dtype)
      end

      def call(array)
        # TODO(sonots): pass device from outside
        device = Chainer::Device.default
        fan_in, fan_out = Chainer::Utils::Initializer.get_fans(array.shape, device: device)
        s = @scale * device.xm::NMath.sqrt(2.0 / fan_in)
        Normal.new(scale: s).(array)
      end
    end
  end
end
