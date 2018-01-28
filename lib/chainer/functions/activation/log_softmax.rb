module Chainer
  module Functions
    module Activation
      def self.logsumexp(x)
        m = x.max(axis: 1, keepdims: true)
        y = x - m
        y = Numo::NMath.exp(y)
        s = y.sum(axis: 1, keepdims: true)
        s = Numo::NMath.log(s)
        m + s
      end

      def self._log_softmax(x)
        log_z = logsumexp(x)
        x - log_z
      end

      # Log-softmax activation function.
      class LogSoftmax < Function
        # Channel-wise log-softmax function.
        #
        # This function computes its logarithm of softmax along the second axis.
        # Let $c = (c_1, c_2, \\dots, c_D)$ be the slice of +x+ along with
        # the second axis. For each slice $c$, it computes the logarithm of
        # the function $f(\c)$ defined as
        #
        # $$
        # f(\c) = { \\exp(\c) \\over \\sum_{ d } \\exp(c_d) }.
        # $$
        #
        # This method is theoretically equivalent to +log(softmax(x))+ but is more
        # stable.
        #
        # @note
        #   +log(softmax(x))+ may cause underflow when +x+ is too small,
        #   because +softmax(x)+ may returns +0+.
        #   +log_softmax+ method is more stable.
        #
        # @param [Chainer::Variable or Numo::NArray] x Input variable. A $n$-dimensional ($n \\geq 2$) float array.
        # @return [Chainer::Variable] Output variable. A $n$-dimensional ($n \\geq 2$) float array, which is the same shape with x.
        #
        # @see Chainer::Functions::Softmax
        #
        # @example
        #   > x = Numo::SFloat[[0, 1, 2], [0, 2, 4]]
        #   => Numo::SFloat#shape=[2,3]
        #   [[0, 1, 2],
        #    [0, 2, 4]]
        #   > F = Chainer::Functions::Activation::LogSoftmax
        #   > F.log_softmax(x).data
        #   => Numo::SFloat#shape=[2,3]
        #   [[-2.40761, -1.40761, -0.407606],
        #    [-4.14293, -2.14293, -0.142932]]
        # @example (T.B.I : F.log, F.softmax)
        #   > F.log_softmax(x).data.nearly_eq(F.log(F.softmax(x)).data).all?)
        #   => true
        #
        def self.log_softmax(x)
          self.new.(x)
        end

        def forward(xs)
          y = Chainer::Functions::Activation._log_softmax(xs[0])
          @x_shape = xs[0].shape
          @x_dtype = xs[0].class
          retain_inputs([])
          retain_outputs([0])
          [y]
        end

        def backward(x, gy)
          y = @output_data[0]
          gx = gy[0] - Numo::NMath.exp(y) * gy[0].sum(axis: 1, keepdims: true)
          [gx]
        end
      end
    end
  end
end
