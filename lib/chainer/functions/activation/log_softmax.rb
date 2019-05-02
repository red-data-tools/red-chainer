module Chainer
  module Functions
    module Activation
      def self.logsumexp(x)
        xm = Chainer.get_array_module(x)
        m = x.max(axis: 1, keepdims: true)
        y = x - m
        y = xm::NMath.exp(y)
        s = y.sum(axis: 1, keepdims: true)
        s = xm::NMath.log(s)
        m + s
      end

      def self._log_softmax(x)
        log_z = logsumexp(x)
        x - log_z
      end

      # Log-softmax activation function.
      class LogSoftmax < FunctionNode
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
        # @param [Chainer::Variable or Numo::NArray or Cumo::NArray] x Input variable. A $n$-dimensional ($n \\geq 2$) float array.
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
          self.new.apply([x]).first
        end

        def forward(xs)
          y = Chainer::Functions::Activation._log_softmax(xs[0])
          @x_shape = xs[0].shape
          @x_dtype = xs[0].class
          retain_outputs([0])
          [y]
        end

        def backward(indexes, gy)
          y = get_retained_outputs.first
          LogSoftmaxGrad.new(@x_shape, @x_dtype).apply([y, gy[0]])
        end
      end

      class LogSoftmaxGrad < FunctionNode
        def initialize(x_shape, x_dtype)
          @x_shape = x_shape
          @x_dtype = x_dtype
        end

        def forward(inputs)
          retain_inputs([0, 1])
          y, gy = inputs

          xm = Chainer.get_array_module(y)
          gx = gy - xm::NMath.exp(y) * gy.sum(axis: 1, keepdims: true)
          [gx]
        end

        def backward(indexes, ggx)
          y, gy = get_retained_inputs
          ret = []
          exp_y = Chainer::Functions::Math::Exp.exp(y)

          if indexes.include?(0)
            gy_sum = Chainer::Functions::Math::Sum.sum(gy, axis: 1, keepdims: true)
            gy_sum = Chainer::Functions::Array::BroadcastTo.broadcast_to(gy_sum, gy.shape)

            g0 = -ggx.first * exp_y * gy_sum
            ret << g0
          end
          if indexes.include?(1)
            a = Chainer::Functions::Math::Sum.sum(ggx.first * exp_y, axis: 1, keepdims: true)
            a = Chainer::Functions::Array::BroadcastTo.broadcast_to(a, gy.shape)
            g1 = ggx.first - a
            ret << g1
          end

          ret
        end
      end
    end
  end
end
