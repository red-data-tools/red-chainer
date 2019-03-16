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

          col = Chainer::Utils::Conv.im2col(x[0], @kh, @kw, @sy, @sx, @ph, @pw, pval: -Float::INFINITY, cover_all: @cover_all)
          n, c, kh, kw, out_h, out_w = col.shape
          col = col.reshape(n , c, kh * kw, out_h, out_w)

          # TODO: numpy.argmax(axis=2)
          d = col.shape[3..-1].reduce(:*) || 1
          dx = col.shape[2..-1].reduce(:*) || 1
          max_index = col.max_index(2)
          @indexes = max_index.flatten.map_with_index { |val, idx| (val - (dx * (idx / d))) / d }.reshape(*max_index.shape)

          y = col.max(axis: 2)
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
          @cover_all = mpool2d.cover_all
          @indexes = mpool2d.indexes
          @in_shape = mpool2d.in_shape
          @in_dtype = mpool2d.in_dtype
          @mpool2d = mpool2d
        end

        def forward(gy)
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
        end

        def forward(x)
          col = Chainer::Utils::Conv.im2col(x[0], @kh, @kw, @sy, @sx, @ph, @pw, pval: -Float::INFINITY, cover_all: @cover_all)
          n, c, kh, kw, out_h, out_w = col.shape
          col = col.reshape(n, c, kh * kw, out_h, out_w)
          col = col.transpose(0, 1, 3, 4, 2).reshape(nil, kh * kw)

          indexes = @indexes.flatten.dup

          # TODO: col = col[numpy.arange(len(indexes)), indexes]
          new_col = col.class.zeros(indexes.size)
          x[0].class.new(indexes.size).seq.each_with_index do |v, i|
            new_col[i] = col[v, indexes[i]]
          end
          col = new_col

          [col.reshape(n, c, out_h, out_w)]
        end
      end
    end
  end
end
