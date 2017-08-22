module Chainer
  module Functions
    module Activation
      class Relu < Function
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
