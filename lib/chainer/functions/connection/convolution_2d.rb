module Chainer
  module Functions
    module Connection
      class Convolution2DFunction < Chainer::FunctionNode
        attr_reader :sy, :sx, :ph, :pw, :cover_all
        # Two-dimensional convolution function.
        # This is an implementation of two-dimensional convolution in ConvNets.
        # It takes three variables: the input image `x`, the filter weight `w`, and the bias vector `b`.
        #
        # a notation for dimensionalities.
        #
        # - :math:`n` is the batch size.
        # - :math:`c_I` and :math:`c_O` are the number of the input and output channels, respectively.
        # - :math:`h_I` and :math:`w_I` are the height and width of the input image, respectively.
        # - :math:`h_K` and :math:`w_K` are the height and width of the filters, respectively.
        # - :math:`h_P` and :math:`w_P` are the height and width of the spatial padding size, respectively.
        #
        # Then the `Convolution2D` function computes correlations between filters and patches of size :math:`(h_K, w_K)` in `x`.
        # Patches are extracted at positions shifted by multiples of `stride` from the first position `(-h_P, -w_P)` for each spatial axis.
        # The right-most (or bottom-most) patches do not run over the padded spatial size.
        # Let :math:`(s_Y, s_X)` be the stride of filter application.
        # Then, the output size :math:`(h_O, w_O)` is determined by the following equations:
        #
        # math:
        #  h_O &= (h_I + 2h_P - h_K) / s_Y + 1,\\\\
        #   w_O &= (w_I + 2w_P - w_K) / s_X + 1.
        # If `cover_all` option is `true`, the filter will cover the all spatial locations.
        # So, if the last stride of filter does not cover the end of spatial locations,
        # an addtional stride will be applied to the end part of spatial locations.
        # In this case, the output size :math:`(h_O, w_O)` is determined by the following equations:
        #
        # math:
        #  h_O &= (h_I + 2h_P - h_K + s_Y - 1) / s_Y + 1,\\\\
        #  w_O &= (w_I + 2w_P - w_K + s_X - 1) / s_X + 1.
        # If the bias vector is given, then it is added to all spatial locations of the output of convolution.
        #
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable of shape :math:`(n, c_I, h_I, w_I)`.
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] w Weight variable of shape :math:`(c_O, c_I, h_K, w_K)`.
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] b Bias variable of length :math:`c_O`
        # @param [Int or 2-D Array] stride Stride of filter applications. `stride=s` and `stride=(s, s)` are equivalent.
        # @param [Int or 2-D Array] pad Spatial padding width for input arrays.
        # @param [Boolean] cover_all If `true`, all spatial locations are convoluted into some output pixels.
        # @return [Chainer::Variable] Output variable of shape :math:`(n, c_O, h_O, w_O)`.
        def self.convolution_2d(x, w, b: nil, stride: 1, pad: 0, cover_all: false)
          func = self.new(stride: stride, pad: pad, cover_all: cover_all)
          if b.nil?
            args = [x, w]
          else
            args = [x, w, b]
          end

          func.apply(args).first
        end

        def initialize(stride: 1, pad: 0, cover_all: false)
          @sy, @sx = stride.is_a?(::Array) ? stride : [stride, stride]
          @ph, @pw = pad.is_a?(::Array) ? pad : [pad, pad]
          @cover_all = cover_all
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x = inputs[0]
          w = inputs[1]
          b = inputs.size == 3 ? inputs[2] : nil

          xm = Chainer.get_array_module(x)
          unless inputs.all? { |i| i.is_a?(xm::NArray) }
            if b.nil?
              raise TypeError, "#{xm}::NArray must not be used together w: #{w.class}, x: #{x.class}"
            else
              raise TypeError, "#{xm}::NArray must not be used together w: #{w.class}, x: #{x.class}, b: #{b.class}"
            end
          end

          if xm == Cumo and Cumo::CUDA::CUDNN.available? and !@cover_all
            return _forward_cudnn(x, w, b)
          end

          kh, kw = w.shape[2..-1]
          col = Chainer::Utils::Conv.im2col(x, kh, kw, @sy, @sx, @ph, @pw, cover_all: @cover_all)
          y = Chainer::Utils::Math.tensordot(col, w, [[1, 2, 3], [1, 2, 3]]).cast_to(x.class)
          y = y.transpose(0, 3, 1, 2) # (N, oC, oH, oW)
          if !b.nil?
            y += b.reshape(1, b.size, 1, 1)
          end

          [y]
        end

        private def _forward_cudnn(x, w, b)
          w = w.cast_to(x.class)
          b = b.cast_to(x.class) if b
          [x.conv(w, b: b, stride: [@sy, @sx], pad: [@ph, @pw])]
        end

        def backward(indexes, grad_outputs)
          x, w = get_retained_inputs
          gy = grad_outputs.first

          ret = []
          if indexes.include?(0)
            xh, xw = x.shape[2..-1]
            gx = Deconvolution2DFunction.deconvolution_2d(gy, w, stride: [@sy, @sx], pad: [@ph, @pw], outsize: [xh, xw])
            ret << gx
          end

          if indexes.include?(1)
            gw = Chainer::Functions::Connection::Convolution2DGradW.new(self).apply([x, gy]).first
            ret << gw
          end

          if indexes.include?(2)
            gb = Chainer::Functions::Math::Sum.sum(gy, axis: [0, 2, 3])
            ret << gb
          end

          ret
        end
      end
    end
  end
end
