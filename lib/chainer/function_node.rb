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
    attr_accessor :rank, :inputs, :outputs

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
    def apply(inputs)
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

    # Computes the output arrays from the input arrays.
    #
    # @param [Array] inputs input array(s)
    # @return [Array] output array(s)
    def forward(inputs)
      raise TypeError, "mustt inputs > 0, inputs size is #{inputs.size}" if inputs.size.zero?
      # TODO GPU
      forward_cpu(inputs)
    end

    # Computes the output arrays from the input Numo::NArray.
    #
    # @param [Array<Numo::NArray>] inputs Numo::NArray objects.
    # @return [Array<Numo::NArray>] Array of output arrays.
    def forward_cpu(inputs)
      raise NotImplementedError
    end

    # Lets specified input variable nodes keep data arrays.
    #
    # By calling this method from `forward`, the function node can specify which inputs are required for backprop.
    # The input variables with retained arrays can be obtained by `get_retained_inputs` from `backward`.
    #
    # Note that **this method must not be called from the outside of forward method.**
    # @param [Integer, Array] indexes Indexes of input variables that the function does not require for backprop.
    def retain_inputs(indexes)
      @input_indexes_to_retain = indexes
    end

    # Lets specified output variable nodes keep data arrays.
    #
    # By calling this method from `forward`, the function node can specify which outputs are required for backprop.
    # If this method is not called, any output variables are not marked to keep the data array at the point of returning from `apply`.
    # The output variables with retained arrays can be obtained by `get_retained_outputs` from `backward`.
    # Note that **this method must not be called from the outside of forward method.**
    # @param [Integer, Array] indexes Indexes of input variables that the function does not require for backprop.
    def retain_outputs(indexes)
      @output_indexes_to_retain = indexes
    end

    # Computes gradients w.r.t. specified inputs given output gradients.
    #
    # This method is used to compute one step of the backpropagation corresponding to the forward computation of this function node.
    # Given the gradients w.r.t. output variables, this method computes the gradients w.r.t. specified input variables.
    # Note that this method does not need to compute any input gradients not specified by `target_input_indexes`
    # It enables the function node to return the input gradients with the full computational history,
    #   in which case it supports *differentiable backpropagation* or *higher-order differentiation*.
    #
    # @param [Array<Integer>] target_indexes Indices of the input variables w.r.t. which the gradients are required.
    #   It is guaranteed that this tuple contains at least one element.
    # @param [Array<Chainer::Variable>] grad_outputs Gradients w.r.t. the output variables.
    #   If the gradient w.r.t. an output variable is not given, the corresponding element is `None`.
    # @return [Array<Chainer::Variable>] Array of Chainer::Variable that represent the gradients.
    def backward(target_indexes, grad_outputs)
      [nil] * target_indexes.size
    end

    # Computes gradients w.r.t. specified inputs and accumulates them.
    #
    # This method provides a way to fuse the backward computation and the gradient accumulations
    #   in the case that the multiple functions are applied to the same variable.
    # Users have to override either of this method or `backward`.
    # It is often simpler to implement `backward` and is recommended if you do not need to provide efficient gradient accumulation.
    #
    # @param [Array<Integer>] target_indexes Indices of the input variables w.r.t. which the gradients are required.
    #   It is guaranteed that this tuple contains at least one element.
    # @param [Array<Chainer::Variable>] grad_outputs Gradients w.r.t. the output variables.
    #   If the gradient w.r.t. an output variable is not given, the corresponding element is `None`.
    # @param [Array<Chainer::Variable>] grad_inputs Gradients w.r.t. the input variables specified by `target_input_indexes`.
    #   These values are computed by other computation paths.
    #   If there is no gradient value existing for the variable, the corresponding element is ``None``.
    # @return [Array<Chainer::Variable>] Array of variables that represent the gradients w.r.t. specified input variables.
    def backward_accumulate(target_input_indexes, grad_outputs, grad_inputs)
      gxs = backward(target_input_indexes, grad_outputs)

      len_gxs = gxs.size
      if len_gxs == @inputs.size
        gxs = target_indexes.map { |i| gxs[i] }
      elsif len_gxs != target_input_indexes.size
        raise ArgumentError, "number of gradients returned by #{impl_name} (#{label}) is incorrect."
      end

      gxs.zip(grad_inputs).map do |gx, g_input|
        if g_input.nil?
          gx
        elsif gx.nil?
          g_input
        else
          gx + g_input
        end
      end
    end

    # Returns a Array of retained input variables.
    #
    # This method is used to retrieve the input variables retained in `forward`.
    #
    # @return [Array] a Array of retained input variables.
    def get_retained_inputs
      @input_indexes_to_retain.map { |index| @inputs[index].get_variable }
    end

    # Returns a Array of retained output variables.
    #
    # This method is used to retrieve the input variables retained in `forward`.
    #
    # @return [Array] a Array of retained input variables.
    def get_retained_outputs
      ret = []
      outputs = @outputs

      new_outputs = outputs.dup
      outputs_modified = false

      @output_indexes_to_retain.zip(@retained_output_data) do |index, data|
        output = outputs[indx].()
        if output.nil?
          output_var = Chainer::Variable.new(data)
          output_var.creator_node = self
          new_outputs[index] = WeakRef.new(output_var)
          outputs_modified = true
        else
          output_var = output.get_variable
        end

        ret << output_var
      end

      if outputs_modified
        @outputs = Array(new_outputs)
      end

      ret
    end

    # Purges in/out nodes and this function node itself from the graph.
    def unchain
      @outputs.each do |y|
        y_ref = y.()
        unless y_ref.nil?
          y_ref.unchain
        end
      end
      @inputs = nil
    end

    private

    def impl_name
      self.class.name
    end

  end
end
