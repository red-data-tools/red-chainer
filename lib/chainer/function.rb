module Chainer
  class Function

    attr_reader :rank, :inputs, :outputs, :retain_after_backward
    attr_accessor :output_data

    def initialize
      @rank = 0
    end

    def call(*inputs)
      inputs = inputs.map do |x|
        if x.kind_of?(Chainer::Variable)
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
      raise if !outputs.is_a? Array

      ret = outputs.map do |y|
        Variable.new(y, requires_grad: requires_grad)
      end

      if Chainer.configuration.enable_backprop
        @rank = inputs.map(&:rank).max || 0

        ret.each { |y| y.creator = self }

        @inputs = inputs.map(&:node)
        @outputs = ret.map { |y| WeakRef.new(y.node) }

        @input_indexes_to_retain = 0...inputs.size if @input_indexes_to_retain.nil?
        @input_indexes_to_retain.each do |index|
          inputs[index].retain_data()
        end
        remove_instance_variable(:@input_indexes_to_retain)

        unless @output_indexes_to_retain.nil?
          @output_indexes_to_retain.each do |index|
            ret[index].retain_data()
          end
          remove_instance_variable(:@output_indexes_to_retain)
        end
      end

      ret.size == 1 ? ret[0] : ret
    end

    def forward(inputs)
      # TODO: GPU branch processing
      forward_cpu(inputs)
    end

    def forward_cpu(inputs)
      raise NotImplementedError
    end

    def backward(inputs, grad_outputs)
      backward_cpu(inputs, grad_outputs)
    end

    def retain_inputs(indexes)
      @input_indexes_to_retain = indexes
    end

    def retain_outputs(indexes, retain_after_backward: false)
      @output_indexes_to_retain = indexes
      if retain_after_backward
        @retain_after_backward = retain_after_backward
      end
    end
  end
end
