module Chainer
  module Functions
    module Connection
      class Convolution2DFunction < Chainer::Function
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
        # @param [Chainer::Variable or Numo::NArray] x Input variable of shape :math:`(n, c_I, h_I, w_I)`.
        # @param [Chainer::Variable or Numo::NArray] w Weight variable of shape :math:`(c_O, c_I, h_K, w_K)`.
        # @param [Chainer::Variable or Numo::NArray] b Bias variable of length :math:`c_O`
        # @param [Int or 2-D Array] stride Stride of filter applications. `stride=s` and `stride=(s, s)` are equivalent.
        # @param [Int or 2-D Array] pad Spatial padding width for input arrays.
        # @param [Boolean] cover_all If `true`, all spatial locations are convoluted into some output pixels.
        # @return [Chainer::Variable] Output variable of shape :math:`(n, c_O, h_O, w_O)`.
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
          y = Chainer::Utils::Math.tensordot(@col, w, [[1, 2, 3], [1, 2, 3]])
          y += b if b
          
          [y.transpose(0, 3, 1, 2)]
        end

        def backward_cpu(inputs, grad_outputs)
          x, w, b = inputs[0], inputs[1], inputs[2]
          gy = grad_outputs[0]
          height, width = x.shape[2..-1]

          gw = Chainer::Utils::Math.tensordot(gy, @col, [[0, 2, 3], [0, 4, 5]])
          gcol = Chainer::Utils::Math.tensordot(w, gy, [0, 1])
          gcol = gcol.transpose(3, 0, 1, 2)
          gx = Chainer::Utils::Conv.col2im_cpu(gcol, @sy, @sx, @ph, @pw, height, width)

          if b.nil?
            [gx, gw]
          else
            gb = gy.sum(axis: [0, 2, 3])
            [gx, gw, gb]
          end
        end
      end
    end
  end
end
