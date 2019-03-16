module Chainer
  module Functions
    module Pooling
      # Base class of pooling function over a set of 2d planes
      class Pooling2D < Chainer::FunctionNode
        attr_reader :kh, :kw, :sy, :sx, :ph, :pw, :cover_all

        def initialize(ksize, stride: nil, pad: 0, cover_all: true)
          if stride.nil?
            stride = ksize
          end

          @kh, @kw = ksize.is_a?(::Array) ? ksize : [ksize, ksize]
          @sy, @sx = stride.is_a?(::Array) ? stride : [stride, stride]
          @ph, @pw = pad.is_a?(::Array) ? pad: [pad, pad]

          @cover_all = cover_all
        end
      end
    end
  end
end
