module Chainer
  module Functions
    module Pooling
      class MaxPooling2D < Pooling2D
        attr_reader :in_shape, :in_dtype, :indexes
        # Spatial max pooling function
        #
        # @param [Chainer::Variable] x Input variable
        # @param [integer || 2D integer array] ksize Size of pooling window
        # @param [integer || 2D integer array] stride Stride of pooling applications
        # @param [integer || 2D integer array] pad Spatial padding width for the input array
        # @param [boolean] cover_all If `true`, all spatial locations are pooled int some output pixels
        # @return [Chainer::Variable] Output variable
        def self.max_pooling_2d(x, ksize, stride: nil, pad: 0, cover_all: true)
          self.new(ksize, stride: stride, pad: pad, cover_all: cover_all).apply([x]).first
        end

        def forward(x)
          @in_shape = x[0].shape
          @in_dtype = x[0].class

          xm = Chainer.get_array_module(x[0])
          if @use_cudnn = (xm == Cumo and Cumo::CUDA::CUDNN.available? and !@cover_all)
            return _forward_cudnn(x[0])
          end

          col = _im2col(x[0])
          @indexes = _compute_indexes(col)

          y = col.max(axis: 2)
          [y]
        end

        def _im2col(x)
          col = Chainer::Utils::Conv.im2col(x, @kh, @kw, @sy, @sx, @ph, @pw, pval: -Float::INFINITY, cover_all: @cover_all)
          n, c, kh, kw, out_h, out_w = col.shape
          col.reshape(n , c, kh * kw, out_h, out_w)
        end

        def _compute_indexes(col)
          # TODO: numpy.argmax(axis=2)
          d = col.shape[3..-1].reduce(:*) || 1
          dx = col.shape[2..-1].reduce(:*) || 1
          max_index = col.max_index(2)
          max_index.flatten.map_with_index { |val, idx| (val - (dx * (idx / d))) / d }.reshape(*max_index.shape)
        end

        private def _forward_cudnn(x)
          retain_inputs([0])
          y = x.max_pool([@kh, @kw], stride: [@sy, @sx], pad: [@ph, @pw])
          retain_outputs([0])
          [y]
        end

        def backward(indexes, gy)
          MaxPooling2DGrad.new(self).apply(gy)
        end
      end

      class MaxPooling2DGrad < FunctionNode
        def initialize(mpool2d)
          @kh = mpool2d.kh
          @kw = mpool2d.kw
          @sy = mpool2d.sy
          @sx = mpool2d.sx
          @ph = mpool2d.ph
          @pw = mpool2d.pw
          @use_cudnn = mpool2d.use_cudnn
          @cover_all = mpool2d.cover_all
          @indexes = mpool2d.indexes
          @in_shape = mpool2d.in_shape
          @in_dtype = mpool2d.in_dtype
          @mpool2d = mpool2d
        end

        def forward(gy)
          if @use_cudnn
            return _forward_cudnn(gy[0])
          end

          n, c, out_h, out_w = gy[0].shape
          h, w  = @in_shape[2..-1]
          kh, kw = @kh, @kw

          gcol = @in_dtype.zeros(n * c * out_h * out_w * kh * kw)

          indexes = @indexes.flatten
          indexes += indexes.class.new((indexes.size * kh * kw) / (kh * kw)).seq(0, kh * kw)

          gcol[indexes] = gy[0].flatten.dup
          gcol = gcol.reshape(n, c, out_h, out_w, kh, kw)
          gcol = gcol.swapaxes(2, 4)
          gcol = gcol.swapaxes(3, 5)

          gx = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, h, w)
          [gx]
        end

        private def _forward_cudnn(gy)
          x = @mpool2d.get_retained_inputs.first.data
          y = @mpool2d.get_retained_outputs.first.data
          gx = x.max_pool_backward(y, gy, [@kh, @kw], stride: [@sy, @sx], pad: [@ph, @pw])
          return [gx]
        end

        def backward(indexes, ggx)
          MaxPooling2DWithIndexes.new(@mpool2d).apply(ggx)
        end
      end

      class MaxPooling2DWithIndexes < FunctionNode
        def initialize(mpool2d)
          @kh = mpool2d.kh
          @kw = mpool2d.kw
          @sy = mpool2d.sy
          @sx = mpool2d.sx
          @ph = mpool2d.ph
          @pw = mpool2d.pw
          @cover_all = mpool2d.cover_all
          @indexes = mpool2d.indexes
          @use_cudnn = mpool2d.use_cudnn
          @mpool2d = mpool2d
        end

        def forward(x)
          col = Chainer::Utils::Conv.im2col(x[0], @kh, @kw, @sy, @sx, @ph, @pw, pval: -Float::INFINITY, cover_all: @cover_all)
          n, c, kh, kw, out_h, out_w = col.shape
          col = col.reshape(n, c, kh * kw, out_h, out_w)
          col = col.transpose(0, 1, 3, 4, 2).reshape(nil, kh * kw)

          if @use_cudnn and @indexes.nil?
            @indexes = _compute_indexes
          end
          indexes = @indexes.flatten.dup

          # col = col[numpy.arange(len(indexes)), indexes]
          col = col[true, indexes].diagonal.dup

          [col.reshape(n, c, out_h, out_w)]
        end

        private def _compute_indexes
          x = @mpool2d.get_retained_inputs.first.data
          col = @mpool2d._im2col(x)
          @mpool2d._compute_indexes(col)
        end
      end
    end
  end
end
