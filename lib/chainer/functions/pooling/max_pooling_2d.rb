module Chainer
  module Functions
    module Pooling
      class MaxPooling2D < Pooling2D
        # Spatial max pooling function
        #
        # @param [Chainer::Variable] x Input variable
        # @param [integer || 2D integer array] Size of pooling window
        # @param [integer || 2D integer array] Stride of pooling applications
        # @param [integer || 2D integer array] Spatial padding width for the input array
        # @param [boolean] If `true`, all spatial locations are pooled int some output pixels
        # @return [Chainer::Variable] Output variable
        def self.max_pooling_2d(x, ksize, stride: nil, pad: 0, cover_all: true)
          self.new(ksize, stride: stride, pad: pad, cover_all: cover_all).(x)
        end

        def forward(x)
          retain_inputs([])
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

        def backward(x, gy)
          n, c, out_h, out_w = gy[0].shape
          h, w  = @in_shape[2..-1]
          kh, kw = @kh, @kw

          gcol = @in_dtype.zeros(n * c * out_h * out_w * kh * kw)

          indexes = @indexes.flatten
          xm = Chainer.get_array_module(x, gy)
          indexes += xm::Int64.new((indexes.size * kh * kw) / (kh * kw)).seq(0, kh * kw)
         
          gcol[indexes] = gy[0].flatten.dup
          gcol = gcol.reshape(n, c, out_h, out_w, kh, kw)
          gcol = gcol.swapaxes(2, 4)
          gcol = gcol.swapaxes(3, 5)

          gx = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, h, w)
          [gx]
        end
      end
    end
  end
end
