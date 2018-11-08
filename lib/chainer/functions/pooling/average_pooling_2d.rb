module Chainer
  module Functions
    module Pooling
      class AveragePooling2D < Pooling2D
        # Spatial average pooling function.
        #
        # This function acts similarly to :class:`Convolution2D`,
        # but it computes the average of input spatial patch for each channel
        # without any parameter instead of computing the inner products.
        # @param [Chainer::Variable] x Input variable.
        # @param [integer] ksize Size of pooling window. `ksize=k` and `ksize=[k, k]` are equivalent.
        # @param [integer] stride Stride of pooling applications. `stride=s` and `stride=[s, s]` are equivalent.
        #                  If `nil` is specified, then it uses same stride as the pooling window size.
        # @param [integer] pad Spatial padding width for the input array. `pad=p` and `pad=[p, p]` are equivalent.
        # @return [Chainer::Variable] Output variable
        def self.average_pooling_2d(x, ksize, stride: nil, pad: 0)
          self.new(ksize, stride: stride, pad: pad, cover_all: false).(x)
        end

        # Average pooling over a set of 2d planes.
        def forward(x)
          retain_inputs([])
          @in_shape = x[0].shape
          @in_dtype = x[0].class

          col = Chainer::Utils::Conv.im2col(x[0], @kh, @kw, @sy, @sx, @ph, @pw)
          y = col.mean(axis: [2, 3])

          [y]
        end

        def backward(x, gy)
          h, w  = @in_shape[2..-1]
          shape = gy[0].shape
          shape.insert(2, 1, 1)
          gcol = gy[0].reshape(*shape).tile(1, 1, @kh, @kw, 1, 1)

          gx = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, h, w)
          gx /= @kh * @kw
          [gx]
        end
      end
    end
  end
end
