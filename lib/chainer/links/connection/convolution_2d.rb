module Chainer
  module Links
    module Connection
      class Convolution2D < ::Chainer::Link
        # Two-dimensional convolutional layer.
        # 
        # This link wraps the :func:`chainer.functions.convolution_2d` function
        # and holds the filter weight and bias vector as parameters.
        # 
        # @param [integer or nil] in_channels Number of channels of input arrays.
        #     If `nil`, parameter initialization will be deferred until the first forward data pass at which time the size will be determined.
        # @param [integer] out_channels Number of channels of output arrays.
        # @param [integer or 2-d int array] ksize Size of filters (a.k.a. kernels).
        # @param [integer or 2-d int array] stride Stride of filter applications.
        # @param [integer or 2-d int array] pad Spatial padding width for input arrays.
        # @param [boolean] nobias If `true`, then this link does not use the bias term.
        # @param [Numo::NArray or Cumo::NArray] initial_w Initial weight value. If `nil`, the default initializer is used.
        # @param [Numo::NArray or Cumo::NArray] initial_bias Initial bias value. If `nil`, the bias is set to 0.
        #
        # Example
        # There are several ways to make a Convolution2D link.
        # Let an input vector `x` be:
        # > x = Numo::DFloat.new(1, 3, 10, 10).seq
        #
        # 1. Give the first three arguments explicitly:
        # > l = Chainer::Links::Connection::Convolution2D.new(3, 7, 5)
        # > y = l.(x)
        # > y.shape
        # [1, 7, 6, 6]
        #
        # 2. Omit `in_channels` or fill it with `nil`:
        #    The below two cases are the same.
        #
        # > l = Chainer::Links::Connection::Convolution2D.new(7, 5)
        # > y = l.(x)
        # > y.shape
        # [1, 7, 6, 6]
        #
        # > l = Chainer::Links::Connection::Convolution2D.new(nil, 7, 5)
        # > y = l.(x)
        # > y.shape
        # [1, 7, 6, 6]
        #
        # When you omit the first argument, you need to specify the other subsequent arguments from `stride` as keyword auguments.
        #
        # > l = Chainer::Links::Connection::Convolution2D.new(7, 5, stride: 1, pad: 0)
        # > y = l.(x)
        # > y.shape
        # [1, 7, 6, 6]
        def initialize(in_channels, out_channels, ksize=nil, stride: 1, pad: 0, nobias: false, initial_w: nil, initial_bias: nil)
          super()
          if ksize.nil?
            out_channels, ksize, in_channels = in_channels, out_channels, nil
          end

          @ksize = ksize
          @stride = stride.is_a?(Array) ? stride : [stride, stride]
          @pad = pad.is_a?(Array) ? pad : [pad, pad]
          @out_channels = out_channels
          
          init_scope do
            w_initializer = Chainer::Initializers.get_initializer(initial_w)
            @w = Chainer::Parameter.new(initializer: w_initializer)
            if in_channels
              initialize_params(in_channels)
            end

            if nobias
              @b = nil
            else
              initial_bias = 0 if initial_bias.nil?
              bias_initializer = Chainer::Initializers.get_initializer(initial_bias)
              @b = Chainer::Parameter.new(initializer: bias_initializer, shape: out_channels)
            end
          end
        end

        # Applies the convolution layer.
        # @param [Chainer::Variable] x Input image.
        # @return [Chainer::Variable] Output of the convolution.
        def call(x)
          initialize_params(x.shape[1]) if @w.data.nil?
          Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(x, @w, b: @b, stride: @stride, pad: @pad)
        end

        private

        def initialize_params(in_channels)
          kh, kw = @ksize.is_a?(Array) ? @ksize : [@ksize, @ksize]
          w_shape = [@out_channels, in_channels, kh, kw]
          @w.init(w_shape)
        end
      end
    end
  end
end


