module Chainer
  module Functions
    module Connection
      class Convolution2DFunction < Chainer::Function
        def self.convolution_2d(x, w, b: nil, stride: 1, pad: 0, cover_all: false)
          func = self.new(stride: stride, pad: pad, cover_all: cover_all)
          if b.nil?
              func.(x, w)
          else
              func.(x, w, b)
          end
        end

        def initialize(stride: 1, pad: 0, cover_all: false)
          @sy, @sx = stride.is_a?(Array) ? stride : [stride, stride]
          @ph, @pw = pad.is_a?(Array) ? pad : [pad, pad]
          @cover_all = cover_all
        end

        def forward_cpu(inputs)
          x = inputs[0]
          w = inputs[1]
          b = inputs.size == 3 ? inputs[2] : nil

          kh, kw = w.shape[2], w.shape[3]

          @col = Chainer::Utils::Conv.im2col_cpu(x, kh, kw, @sy, @sx, @ph, @pw, cover_all: @cover_all)
          col_shape = @col.shape

          # TODO: numpy.tensordot
          y = @col.class.zeros(w.shape[0] ,col_shape[0], col_shape[4], col_shape[5])
          w.shape[0].times do |n|
            y[n, nil, nil, nil] = @col.transpose(0, 4, 5, 1, 2, 3).mulsum(w[n, nil, nil, nil], 3, 4, 5)
          end
          
          if b
            y.shape[0].times do |n|
              y[n, nil, nil, nil] += b[n]
            end
          end
          
          [y.transpose(1, 0, 2, 3)]
        end
      end
    end
  end
end
