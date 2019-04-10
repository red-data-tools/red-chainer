module Chainer
  module Functions
    module Connection
      class Convolution2DGradW < Chainer::FunctionNode
        def initialize(conv2d)
          w_node = conv2d.inputs[1]

          @kh, @kw = w_node.shape[2..-1]
          @sy = conv2d.sy
          @sx = conv2d.sx
          @ph = conv2d.ph
          @pw = conv2d.pw
          @cover_all = conv2d.cover_all
          @w_dtype = w_node.dtype
          @w_shape = w_node.shape
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x, gy = inputs

          xm = Chainer.get_array_module(x, gy)
          if xm == Cumo and Chainer::CUDA.cudnn_enabled? and !@cover_all
            return _forward_cudnn(x, gy)
          end

          col = Chainer::Utils::Conv.im2col(x, @kh, @kw, @sy, @sx, @ph, @pw, cover_all: @cover_all)
          gw = Chainer::Utils::Math.tensordot(gy, col, [[0, 2, 3], [0, 4, 5]]).cast_to(@w_dtype)
          [gw]
        end

        private def _forward_cudnn(x, gy)
          gy = gy.cast_to(x.class)
          [x.conv_grad_w(gy, @w_shape, stride: [@sy, @sx], pad: [@ph, @pw])]
        end

        def backward(indexes, grad_outputs)
          x, gy = get_retained_inputs
          ggw = grad_outputs.first

          ret = []
          if indexes.include?(0)
            xh, xw = x.shape[2..-1]
            gx = Deconvolution2DFunction.deconvolution_2d(gy, ggw, stride: [@sy, @sx], pad: [@ph, @pw], outsize: [xh, xw])
            ret << gx
          end

          if indexes.include?(1)
            ggy = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(x, ggw, stride: [@sy, @sx], pad: [@ph, @pw], cover_all: @cover_all)
            ret << ggy
          end

          ret
        end
      end
    end
  end
end

