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

    # A tuple of the retained output arrays.
    # This property is mainly used by $Function$. Users basically do
    # not have to use this property; use $get_retained_outputs$ instead.
    def output_data
      raise RuntimeError, 'retained output data is gone' if @retained_output_data.nil?
      out_data = [nil] * @outputs.size
      @output_indexes_to_retain.zip(@retained_output_data).each do |index, data|
        out_data[index] = data
      end
      out_data
    end

    # Computes output variables and grows the computational graph.
    #
    # Basic behavior is expressed in the documentation of `FunctionNode`.
    # @param [Chainer::Variable, Numo::NArray] inputs If the element is an Numo::NArray,
    #   it is automatically wrapped with `Chainer::Variable`.
    # @return [Array<Chainer::Variable>] A tuple of output `Chainer::Variable` objectts.
    def apply(inputs)
      input_vars = inputs.map { |x| Chainer::Variable.as_variable(x) }
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
        gxs = target_input_indexes.map { |i| gxs[i] }
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
        output =  outputs[index].__getobj__
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

  def self.grad(outputs, inputs, grad_outputs: nil, grad_inputs: nil, set_grad: false, retain_grad: false, enable_double_backprop: false)
    # The implementation consists of three steps.

    # 1. Backward enumeration: all the nodes reachable backward from the output
    #    nodes are enumerated. The forward direction links are collected in
    #    this step. Note that the variable nodes whose requires_grad is false
    #    are ignored and their creators are not searched.
    candidate_funcs = outputs.map(&:creator_node).compact
    visited_funcs = Set.new
    forward_graph = {}

    while func = candidate_funcs.pop
      next if visited_funcs.include?(func)
      visited_funcs.add(func)

      func.inputs.each do |x|
        next unless x.requires_grad
        forward_graph[x] = [] if forward_graph[x].nil?
        forward_graph[x] << func
        creator = x.creator_node
        if creator && !visited_funcs.include?(creator)
          candidate_funcs << creator
        end
      end
    end

    # 2. Forward enumeration: all the nodes in the subgraph reachable from the
    #    input nodes are enumerated. The extracted (sub-)subgraph is the union
    #    of all paths that backpropagation will visit.
    candidate_vars = inputs.map(&:node)
    visited_funcs = Set.new
    grad_required = Set.new
    while x = candidate_vars.pop
      grad_required.add(x)
      forward_graph[x].each do |func|
        next if visited_funcs.include?(func)
        visited_funcs.add(func)
        func.outputs.each do |y_ref|
          y = y_ref.__getobj__
          if y && forward_graph[y]
            candidate_vars << y
          end
        end
      end
    end

    # 3. Backpropagation: the backpropagation is executed along the
    #    (sub-)subgraph. It uses the topological order of the subgraph which is
    #    induced by the reversed order of function applications ("rank").
    grads = {}  # mapping from variable nodes to their gradients

    # Initialize the gradient mapping.
    grad_outputs = [nil] * outputs.size if grad_outputs.nil?
    outputs.zip(grad_outputs).each do |y, gy|
      if gy.nil?
        gy_data = y.data.new_ones
        gy = Chainer::Variable.new(gy_data, requires_grad: false)
      end

      grads[y.node] = gy
    end

    unless grad_inputs.nil?
      inputs.zip(grad_inputs).each do |x, gx|
        grads[x.node] = gx unless gx.nil?
      end
    end

    # Backprop implementation. It edits grads which will only contain the
    # gradients w.r.t. the inputs.
    old_enable_backprop = Chainer.configuration.enable_backprop
    Chainer.configuration.enable_backprop = enable_double_backprop
    backprop(outputs, inputs, grad_required, retain_grad, grads)
    Chainer.configuration.enable_backprop = old_enable_backprop

    # Extract the gradients w.r.t. the inputs and return them.
    ret = inputs.map { |x| grads[x.node] }
    if set_grad
      inputs.zip(ret).each do |x, gx|
        x.grad_var = gx
      end
    end

    ret
  end

  def self.backprop(outputs, inputs, grad_required, retain_grad, grads)
    candidate_funcs = []
    visited_funcs = Set.new

    push_candidate = -> (func) do
      return if visited_funcs.include?(func)

      # Negate since heapq is min-heap
      # The second element is used to make each item unique
      visited_funcs.add(func)
      candidate_funcs << func
      candidate_funcs.sort_by! { |f| -f.rank }
    end

    pop_candidate = -> () do
      candidate_funcs.pop
    end

    outputs.each do |y|
      creator = y.creator_node
      next if creator.nil?
      push_candidate.(creator)
    end

    input_nodes = Set.new(inputs.map(&:node))

    while func = pop_candidate.()
      # Collect the gradients w.r.t. the outputs
      gys = []

      func.outputs.each do |y_ref|
        y = y_ref.__getobj__
        if y.nil?
          gys << nil
          next
        end
        gys << grads[y]
      end

      # Collect the gradients w.r.t. the inputs
      #
      # Note (Tokui): when the same variable is passed multiple times as
      # inputs in the same function (e.g. an expression like f(x, x)), the
      # current implementation passes None as the current gradient w.r.t.
      # such an input except for the first one (i.e., it builds gxs like
      # (gx, None) where gx is the current gradient w.r.t. x).
      gxs = []
      input_indexes = []
      selected_inputs = Set.new
      func.inputs.each_with_index do |x, i|
        next unless grad_required.include?(x)

        input_indexes << i
        if selected_inputs.include?(x)
          gxs << nil
        else
          gxs << grads[x]
          selected_inputs.add(x)
        end
      end

      next if input_indexes.empty?

      # Do backward
      new_gxs = func.backward_accumulate(input_indexes, gys, gxs)

      # Delete output gradients that are not required to return
      func.outputs.each do |y_ref|
        y = y_ref.__getobj__
        if y && grads[y] && !input_nodes.include?(y)
          grads.delete(y)
        end
      end

      # Update grads
      selected_inputs = Set.new
      input_indexes.zip(new_gxs).each do |i, g|
        next if g.nil?

        node = func.inputs[i]
        if selected_inputs.include?(node)
          # Accumulate the duplicated gradients here
          cur_gx = grads[node]
          if cur_gx
            g = g + cur_gx
          end
        else
          selected_inputs.add(node)
        end

        grads[node] = g

        if retain_grad
          v = node.get_variable
          if v
            v.grad_var = g
          end
        end

        creator = node.creator_node
        if creator
          push_candidate.(creator)
        end
      end
    end
  end
  private_class_method :backprop
end
