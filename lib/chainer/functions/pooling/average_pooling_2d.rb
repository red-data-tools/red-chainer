module Chainer
  module Functions
    module Pooling
      class AveragePooling2D < Pooling2D
        attr_reader :in_shape, :in_dtype

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
          self.new(ksize, stride: stride, pad: pad, cover_all: false).apply([x])[0]
        end

        # Average pooling over a set of 2d planes.
        def forward(x)
          @in_shape = x[0].shape
          @in_dtype = x[0].class

          xm = Chainer.get_array_module(x[0])
          if @use_cudnn = (xm == Cumo and Cumo::CUDA::CUDNN.available? and !@cover_all)
            return _forward_cudnn(x[0])
          end

          col = Chainer::Utils::Conv.im2col(x[0], @kh, @kw, @sy, @sx, @ph, @pw)
          y = col.mean(axis: [2, 3])

          [y]
        end

        private def _forward_cudnn(x)
          retain_inputs([0])
          y = x.avg_pool([@kh, @kw], stride: [@sy, @sx], pad: [@ph, @pw], pad_value: 0)
          retain_outputs([0])
          [y]
        end

        def backward(indexes, gy)
          AveragePooling2DGrad.new(self).apply(gy)
        end
      end

      class AveragePooling2DGrad < FunctionNode
        def initialize(apool2d)
          @kh = apool2d.kh
          @kw = apool2d.kw
          @sy = apool2d.sy
          @sx = apool2d.sx
          @ph = apool2d.ph
          @pw = apool2d.pw
          @use_cudnn = apool2d.use_cudnn
          @in_shape = apool2d.in_shape
          @in_dtype = apool2d.in_dtype
          @apool2d = apool2d
        end

        def forward(gy)
          if @use_cudnn
            return _forward_cudnn(gy[0])
          end

          h, w  = @in_shape[2..-1]
          shape = gy[0].shape
          shape.insert(2, 1, 1)
          gcol = gy[0].reshape(*shape).tile(1, 1, @kh, @kw, 1, 1)

          gx = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, h, w)
          gx /= @kh * @kw
          [gx]
        end

        private def _forward_cudnn(gy)
          x = @apool2d.get_retained_inputs.first.data
          y = @apool2d.get_retained_outputs.first.data
          gx = x.avg_pool_backward(y, gy, [@kh, @kw], stride: [@sy, @sx], pad: [@ph, @pw], pad_value: 0)
          return [gx]
        end

        def backward(indexes, grad_outputs)
          AveragePooling2D.new([@kh, @kw], stride: [@sy, @sx], pad: [@ph, @pw], cover_all: false).apply(grad_outputs)
        end
      end
    end
  end
end
