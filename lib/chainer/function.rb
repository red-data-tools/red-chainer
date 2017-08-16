module Chainer
  class Function
    def call(*inputs)
      inputs = inputs.map do |x|
        if x.instance_of?(Variable)
          x
        else
          Variable.new(x, requires_grad: false)
        end
      end

      in_data = inputs.map(&:data)
      requires_grad = inputs.any?(&:requires_grad)

      @input_indexes_to_retain = nil
      @output_indexes_to_retain = nil
      outputs = forward(in_data)

      ret = outputs.map do |y|
        Variable.new(y, requires_grad: requires_grad)
      end
      
      if ret.size == 1
        ret[0]
      else
        ret
      end
    end

    def forward(inputs)
      # TODO: GPU branch processing
      forward_cpu(inputs)
    end

    def forward_cpu(inputs)
      raise NotImplementedError
    end

    def retain_inputs(indexes)
      @input_indexes_to_retain = indexes
    end
  end
end
