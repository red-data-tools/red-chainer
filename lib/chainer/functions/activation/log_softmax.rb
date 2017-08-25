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

      def self.log_softmax(x)
        log_z = logsumexp(x)
        x - log_z
      end

      class LogSoftmax < Function
        def self.relu(x)
          self.new.(x)
        end

        def forward_cpu(x)
          retain_inputs([])
          retain_outputs([0])
          x[0][x[0]<=0] = 0
          [Utils::Array.force_array(x[0])] 
        end

        def backward_cpu(x, gy)
          y = output_data[0]
          [Utils::Array.force_array(gy[0] * (y > 0))]
        end
      end
    end
  end
end
