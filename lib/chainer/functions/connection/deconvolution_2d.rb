module Chainer
  module Functions
    module Connection
      class Deconvolution2DFunction < Chainer::FunctionNode
        attr_reader :sy, :sx, :ph, :pw, :cover_all

        # Two dimensional deconvolution function.
        #
        # This is an implementation of two-dimensional deconvolution.
        # In most of deep learning frameworks and papers,
        # this function is called <b>transposed convolution</b>.
        # But because of historical reasons (e.g. paper by Ziller Deconvolutional Networks) and backward compatibility,
        # this function is called +deconvolution+ in Chainer.
        #
        # It takes three variables: input image +x+,
        # the filter weight +W+, and the bias vector +b+.
        #
        # - $n$ is the batch size.
        # - $c_I$ and $c_O$ are the number of the input and output channels, respectively.
        # - $h_I$ and $w_I$ are the height and width of the input image, respectively.
        # - $h_K$ and $w_K$ are the height and width of the filters, respectively.
        # - $h_P$ and $w_P$ are the height and width of the spatial padding size, respectively.
        #
        # Let $(s_Y, s_X)$ be the stride of filter application.
        # Then, the output size $(h_O, w_O)$ is estimated by the following equations:
        #
        # $
        # h_O &= s_Y (h_I - 1) + h_K - 2h_P,
        # w_O &= s_X (w_I - 1) + w_K - 2w_P.
        # $
        #
        # @param [Chainer::Variable or Numo::NArray] x Input variable of shape $(n, c_I, h_I, w_I)$.
        # @param [Chainer::Variable or Numo::NArray] w Weight variable of shape $(c_I, c_O, h_K, w_K)$.
        # @param [Chainer::Variable or Numo::NArray] b Bias variable of length $c_O$ (optional).
        # @param [integer or Array<integer>] stride Stride of filter applications. +stride=s+ and +stride=[s, s]+ are equivalent.
        # @param [integer or Array<integer>] pad Spatial padding width for input arrays. +pad=p+ and +pad=[p, p]+ are equivalent.
        # @param [integer or Arrat<integer>] outsize Expected output size of deconvolutional operation.
        #   It should be pair of height and width $(h_O, w_O)$.
        #   Default value is +nil+ and the outsize is estimated by input size, stride and pad.
        # @return [Chainer::Variable] Output variable of shape $(n, c_O, h_O, w_O)$.
        #
        # Example
        # > n = 10
        # > c_i, c_o = 1, 3
        # > h_i, w_i = 5, 10
        # > h_k, w_k = 10, 10
        # > h_p, w_p = 5, 5
        # > x = Numo::DFloat.new(n, c_i, h_i, w_i).rand
        # > x.shape
        # => [10, 1, 5, 10]
        # > w = Numo::DFloat.new(c_i, c_o, h_k, w_k).rand
        # > w.shape
        # => [1, 3, 10, 10]
        # > b = Numo::DFloat.new(c_o).rand
        # > b.shape
        # => [3]
        # > s_y, s_x = 5, 5
        # > y = Chainer::Functions::Connection::Deconvolution2DFunction.deconvolution_2d(x, w, b: b, stride: [s_y, s_x], pad: [h_p, w_p])
        # > y.shape
        # => [10, 3, 20, 45]
        # > h_o = s_y * (h_i - 1) + h_k - 2 * h_p
        # > w_o = s_x * (w_i - 1) + w_k - 2 * w_p
        # > y.shape == [n, c_o, h_o, w_o]
        # => true
        def self.deconvolution_2d(x, w, b: nil, stride: 1, pad: 0, outsize: nil)
          func = Deconvolution2DFunction.new(stride: stride, pad: pad, outsize: outsize)
          if b.nil?
            args = x, w
          else
            args = x, w, b
          end
          func.apply(args).first
        end

        def initialize(stride: 1, pad: 0, outsize: nil)
          @cover_all = nil

          @sy, @sx = stride.is_a?(::Array) ? stride : [stride, stride]
          @ph, @pw = pad.is_a?(::Array) ? pad : [pad, pad]
          @outh, @outw = outsize.nil? ? [nil, nil] : outsize
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x, w = inputs[0...2]
          b = inputs.size == 3 ? inputs[2] : nil

          xm = Chainer.get_array_module(x)
          unless inputs.all? { |i| i.is_a?(xm::NArray) }
            if b.nil?
              raise TypeError, "#{xm}::NArray must not be used together w: #{w.class}, x: #{x.class}"
            else
              raise TypeError, "#{xm}::NArray must not be used together w: #{w.class}, x: #{x.class}, b: #{b.class}"
            end
          end

          kh, kw = w.shape[2..-1]
          _, _, x_h, x_w = x.shape

          gcol = Chainer::Utils::Math.tensordot(w, x, [0, 1]).cast_to(x.class)
          # - k, m, n: shape of out_channel
          # - b: number of inputs
          # - h, w: height and width of kernels
          # k, m, n, b, h, w -> b, k, m, n, h, w
          gcol = gcol.transpose(3, 0, 1, 2, 4, 5)

          if @outh.nil?
            @outh = Chainer::Utils::Conv.get_deconv_outsize(x_h, kh, @sy, @ph)
            raise TypeError, 'Height in the output should be positive.' if @outh <= 0
          end
          if @outw.nil?
            @outw = Chainer::Utils::Conv.get_deconv_outsize(x_w, kw, @sx, @pw)
            raise TypeError, 'Width in the output should be positive.' if @outw <= 0
          end

          y = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, @outh, @outw)
          if !b.nil?
            y += b.reshape(1, b.size, 1, 1)
          end
          [y]
        end

        def backward(indexes, grad_outputs)
          x, w = get_retained_inputs
          gy = grad_outputs.first

          ret = []

          if indexes.include?(0)
            set_cover_all(x, w) if @cover_all.nil?
            gw = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(gy, w, stride: [@sy, @sx], pad: [@ph, @pw], cover_all: @cover_all)
            ret << gw
          end

          if indexes.include?(1)
            set_cover_all(x, w) if @cover_all.nil?
            gw = Chainer::Functions::Connection::Convolution2DGradW.new(self).apply([gy, x]).first
            ret << gw
          end

          if indexes.include?(2)
            gb = Chainer::Functions::Math::Sum.sum(gy, axis: [0, 2, 3])
            ret << gb
          end

          ret
        end

        private

        def set_cover_all(x, w)
          in_h, in_w = x.shape[2..-1]
          kh, kw = w.shape[2..-1]

          @cover_all = in_h != Chainer::Utils::Conv.get_conv_outsize(@outh, kh, @sy, @ph) || in_w != Chainer::Utils::Conv.get_conv_outsize(@outw, kw, @sx, @pw)
        end
      end
    end
  end
end
