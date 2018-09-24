# Function node of the computational graph.
# FunctionNode is a class representing a node in a computational graph.
# The node corresponds to an application of a differentiable function to input variables.
# When a differentiable function is applied to `Chainer::Variable` objects,
# it creates an instance of FunctionNode implementation and calls its `apply` method.
# The `apply` method basically does the following three things.
#   1. Adding an edge from the function node to the variable node corresponding to each input.
#      The node of each input is extracted by `Chainer::`Variable.node`.
#   2. Computing the output arrays of the function.
#   3. Creating a :class:`Variable` object for each output array and
#      adding an edge from the node of the variable to the function node.
# The output variables are then returned.
module Chainer
  class FunctionNode
    def initialize
      @rank = 0
      @inputs = nil
      @outputs = nil

      @retained_output_data = nil
      @input_indexes_to_retain = nil
      @output_indexes_to_retain = nil
    end

    # Short text that represents the function.
    #
    # The default implementation returns its type name.
    # Each function should override it to give more information.
    def label
      self.class.name
    end

    # Computes output variables and grows the computational graph.
    #
    # Basic behavior is expressed in the documentation of `FunctionNode`.
    # @param [Chainer::Variable, Numo::NArray] inputs If the element is an Numo::NArray,
    #   it is automatically wrapped with `Chainer::Variable`.
    # @return [Array<Chainer::Variable>] A tuple of output `Chainer::Variable` objectts.
    def apply(*inputs)
      input_vars = inputs.map { |x| x.is_a?(Chainer::Variable) ? x : Chainer::Variable.new(x, requires_grad: false) }
      in_data = input_vars.map(&:data)
      requires_grad = input_vars.map(&:requires_grad).any?

      # Forward propagation
      @input_indexes_to_retain = nil
      @output_indexes_to_retain = nil
      outputs = forward(in_data)
      raise TypeError, "#{outputs.class} not Array" unless outputs.is_a?(Array)

      ret = outputs.map { |y| Chainer::Variable.new(y, requires_grad: requires_grad) }

      if Chainer.configuration.enable_backprop
        # Topological ordering
        @rank = input_vars.size > 0 ? input_vars.map(&:rank).max : 0

        # Add backward edges
        ret.each { |y| y.creator_node = self }
        @inputs = input_vars.map(&:node)
        # Add forward edges (must be weak references)
        @outputs = ret.map { |y| WeakRef.new(y.node) }

        unless @input_indexes_to_retain.nil?
          @input_indexes_to_retain.each do |index|
            input_vars[index].retain_data
          end
        end

        unless @output_indexes_to_retain.nil?
          retained_data = []
          @output_indexes_to_retain.each do |index|
            ret[index].retain_data
            retained_data << outputs[index]
          end
          @retained_output_data = Array(retained_data)
        end
      end

      ret
    end


    private

    def impl_name
      self.class.name
    end

  end
end
