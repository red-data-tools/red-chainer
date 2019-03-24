module Chainer
  class VariableNode
    attr_reader :dtype, :shape, :data
    attr_accessor :name, :requires_grad, :variable, :creator_node, :rank, :old_style_grad_generator

    def initialize(variable: , name:)
      @variable = WeakRef.new(variable)
      @creator_node = nil
      @data = nil
      @rank = 0
      @name = name
      @requires_grad = variable.requires_grad

      @old_style_grad_generator = nil

      set_data_type(variable.data)
    end

    def creator
      node = @creator_node
      if node.nil?
        return nil
      end

      if node.is_a?(Chainer::FunctionAdapter)
        return node.function
      end
      node
    end

    def creator=(func)
      self.creator_node = func
    end

    def creator_node=(func)
      func = func.node if func.is_a?(Chainer::Function)
      @creator_node = func
      unless func.nil?
        @rank = func.rank + 1
      end
    end

    def data=(data)
      @data = data
      set_data_type(data)
    end

    # Gradient array of the corresponding variable.
    def grad
      var = get_variable
      var.nil? ? nil : var.grad
    end

    # Gradient variable of the corresponding variable.<Paste>
    def grad_var
			var  = get_variable
      var.nil? ? nil : var.grad_var
    end

    def label
      if @shape.nil? || @shape.empty?
        @dtype.to_s
      else
        "(#{@shape.join(', ')}), #{@dtype.to_s}"
      end
    end

    # Returns the corresponding :class:`Variable` object.
    #
    # @return [Chainer::Variable] The variable object that refers this node.
    def get_variable
      var = @variable
      # workaround: check weakref_alive?, because weakref sometimes delegates references by GC
      return var.__getobj__ if !var.nil? && var.weakref_alive?

      var = Chainer::Variable.new(@data, name: @name, requires_grad: @requires_grad)
      var.node = self
      var
    end

    def set_creator(creator)
      self.creator = creator
    end

    # Sets a `FunctionNode` object that created this node.
    #
    # @params [Chainer::FunctionNode] Function node that has this variable as an output.
    def set_creator_node(creator_node)
      self.creator_node = creator_node
    end

    def unchain
      self.creator_node = nil
    end

    def retain_data
      if @variable.nil?
        raise "cannot retain variable data: the variable has been already released"
      else
        @variable.data
      end
    end

    def set_data_type(data)
      if data.nil?
        @dtype = nil
        @shape = nil
      else
        @dtype = data.class
        @shape = data.shape
      end
    end

    def set_grad_with_check(g, func, var)
      Utils::Variable.check_grad_type(func, var, g)
      @grad = g
    end

    def check_old_style_gradient
      if @old_style_grad_generator
        raise RuntimeError, "cannot twice-differentiate an old style Function #{@old_style_grad_generator}"
      end
    end
  end
end
