require 'chainer/function_node'
module Chainer
  class Function

    attr_reader :rank, :inputs, :outputs, :retain_after_backward
    attr_accessor :output_data, :owned_node

    def initialize
      @rank = 0
    end

    def call(*inputs)
      node = self.node

      node.function = self
      node.weak_function = nil
      @node = WeakRef.new(node)
      @owned_node = nil

      ret = node.apply(inputs)

      ret.size == 1 ? ret[0] : ret
    end

    def inputs
      @node.inputs
    end

    def outputs
      @node.outputs
    end

    def node
      noderef = @node
      nd = noderef ? noderef.__getobj__ : @owned_node
      return nd if nd

      nd = FunctionAdapter.new(self)
      @owned_node = nd
      nd
    end

    def output_data
      node.output_data
    end

    def rank
      @node.rank
    end

    def label
      self.class.to_s
    end

    def forward(inputs)
      xm = Chainer.get_array_module(*inputs)
      if xm == Cumo
        forward_gpu(inputs)
      else
        forward_cpu(inputs)
      end
    end

    def forward_cpu(inputs)
      raise NotImplementedError
    end

    def forward_gpu(inputs)
      raise NotImplementedError
    end

    def backward(inputs, grad_outputs)
      xm = Chainer.get_array_module(*(inputs + grad_outputs))
      if xm == Cumo
        backward_gpu(inputs, grad_outputs)
      else
        backward_cpu(inputs, grad_outputs)
      end
    end

    def backward_cpu(inputs, grad_outputs)
      return [nil] * inputs.size
    end

    def backward_gpu(inputs, grad_outputs)
      return [nil] * inputs.size
    end

    def retain_inputs(indexes)
      @input_indexes_to_retain = indexes
    end

    def retain_outputs(indexes, retain_after_backward: false)
      node.retain_outputs(indexes)
    end
  end

  class FunctionAdapter < ::Chainer::FunctionNode
    attr_accessor :function, :weak_function

    def initialize(function)
      super()
      @weak_function = WeakRef.new(function)
      function.owned_node = self
    end

    def function
      func = @function
      return func if func

      weak_func = @weak_function
      weak_func.__getobj__
    end

    def label
      @function.label
    end

    def forward(inputs)
      retain_inputs(inputs.size.times.to_a)
      @function.forward(inputs)
    end

    def backward(target_input_indexes, grad_outputs)
      in_data = @inputs.map { |input| input.data }
      grad_out_data = grad_outputs.map { |grad| grad.nil? ? nil : grad.data }

      gxs = @function.backward(in_data, grad_out_data)
      ret = []
      target_input_indexes.each do |i|
        if gxs[i].nil?
          g = nil
        else
          g = Chainer::Variable.new(gxs[i])
          g.node.old_style_grad_generator = @function.label
        end
        ret << g
      end

      ret
    end

    # Purges in/out nodes and this function itself from the graph.
    #
    # This method is called from :meth:`Variable.unchain_backward` method.
    def unchain
      @outputs.each do |y|
        y.unchain if y.weakref_alive?
      end
      @inputs = nil
    end
  end
end
