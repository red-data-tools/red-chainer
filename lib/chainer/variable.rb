module Chainer
  class Variable
    attr_accessor :requires_grad, :node

    # Converts an array or a variable into +Chainer::Variable+.
    # This is a convenient function to get a +Chainer::Variable+ object
    # transparently from a raw array or a variable.
    # Note: that this function should only be used for type consistency
    # (i.e. to enforce the return value of an API having type +Chainer::Variable+).
    # The +Chianer::Variable.requires_grad+ flag is kept as is; if +obj+ is a raw array,
    # the newly created variable has +requires_grad = false+.
    # In order to make a variable w.r.t. which you want to compute the gradient,
    # you should use $Chainer::Variable$ directly.
    #
    # @param [Numo::NArray or Chainer::Variable] obj An array or a variable that you want to convert to $Chainer::Variable$.
    # @return [Chainer::Variable] A variable converted from +obj+. If +obj+ is a raw array,
    #   this is a new +Chianer::Variable+ object that wraps the array. If +obj+ is already a +Chainer::Variable+ object, this function returns +obj+ as is.
    def self.as_variable(obj)
      return obj if obj.kind_of?(Chainer::Variable)
      # TODO if obj is_backprop_required is true, set requires_grad = true
      self.new(obj, requires_grad: false)
    end

    def initialize(data=nil, name: nil, grad: nil, requires_grad: true)
      unless data.nil? || Chainer.array?(data)
        raise TypeError, "Numo::NArray or Cumo::NArray are expected."
      end

      @data = data
      @grad = grad
      @requires_grad = requires_grad
      @node = VariableNode.new(variable: self, name: name)
      @grad_var = grad.nil? ? nil : Chainer::Variable.new(grad)
    end

    def inspect
      {data: @data.inspect, grad: @grad.inspect, requires_grad: @requires_grad.inspect}.inspect
    end

    def data
      return @data
    end
    alias_method :array, :data

    def data=(d)
      @data = d
      @node.set_data_type(d)
    end
    alias_method :array=, :data=

    def name
      return @node.name
    end

    def name=(n)
      @node.name = n
    end

    def label
      @node.label
    end

    # deprecated FunctionNode
    def creator
      @node.creator
    end

    def creator=(func)
      @node.creator = func
    end

    def creator_node
      @node.creator_node
    end

    def creator_node=(func)
      @node.creator_node = func
    end

    def grad
      gv = @grad_var
      gv.nil? ? nil : gv.data
    end

    def grad=(g)
      self.grad_var = g.nil? ? nil : Chainer::Variable.new(g)
    end

    def grad_var
      @grad_var
    end

    def grad_var=(g)
      Utils::Variable.check_grad_type(nil, self, g.data) unless g.nil?
      @grad_var = g
    end

    def shape
      self.data.shape
    end

    def ndim
      self.data.ndim
    end

    def size
      self.data.size
    end

    def dtype
      self.data.class
    end

    def rank
      @node.rank
    end

    def transpose
      Chainer::Functions::Array::Transpose.transpose(self)
    end

    def reshape(*shape)
      if shape.size == 1 && shape[0].kind_of?(::Aray)
        shape = shape[0]
      end
      Chainer::Functions::Array::Reshape.reshape(self, shape)
    end

    # Clears the gradient array.
    def cleargrad
      @grad_var = nil
    end

    # Notifies the variable that the given node is its creator.
    #
    # @param [Chainer::FunctionNode] fnode node that has this variable as an output.
    def set_creator_node(fnode)
      @node.set_creator_node(fnode)
    end

    def backward(retain_grad: false, enable_double_backprop: true)
      old_enable_backprop = Chainer.configuration.enable_backprop
      Chainer.configuration.enable_backprop = enable_double_backprop
      _backward_main(retain_grad)
      Chainer.configuration.enable_backprop = old_enable_backprop
    end

    def _backward_main(retain_grad)
      node.check_old_style_gradient
      return if self.creator_node.nil?

      seen_set = Set.new
      grads = {}
      if self.data.size == 1 && self.grad_var.nil?
        self.grad = self.data.new_ones
      end
      grads[self.node] = self.grad_var

      funcs = [self.creator_node]
      seen_set.add(self.creator_node)

      while func = funcs.shift
        inputs = func.inputs
        target_input_indexes = inputs.each_with_index.map { |x, i| i if x.requires_grad }.compact
        next if target_input_indexes.empty?
        outputs = func.outputs.map(&:__getobj__)

        in_data = inputs.map(&:data)
        out_grad = outputs.map do |y|
          next nil if y.nil?
          next grads[y] unless grads[y].nil?
          y.grad_var
        end
        out_grad_data = out_grad.map { |g| g.nil? ? g : g.data }

        # Collect the current input gradients.
        #
        # When the same variable is passed to multiple input slots (e.g. an expression like +f(x, x)+),
        # it makes the gradient accumulation complicated since the back-propagated gradients w.r.t.
        # the first and second argument should be accumulated to the current gradient w.r.t. the same variable.
        # In this case, the current implementation passes the current gradient only to the first occurrence of the variable
        # in the input tuple and passes +nil+ to the rest of the occurrences.
        # For example, when the input variables are +(x, x)+,
        # the input gradient passed to the +backward_accumulate+ method is +(gx, nil)+ where +gx+ is the current gradient of ++x++.
        # See also the docstring of +FunctionNode.backward_accumulate+.
        target_inputs = target_input_indexes.map { |i| inputs[i] }
        in_grad = []
        target_input_indexes.each_with_index do |index_i, i|
          x = inputs[index_i]
          if target_inputs[0...i].include?(x)
            gx = nil
          elsif grads[x]
						gx = grads[x]
          elsif x.creator_node.nil?
            gx = x.grad_var
          else
            gx = nil
          end
          in_grad << gx
        end

        gxs = func.backward_accumulate(target_input_indexes, out_grad, in_grad)
        raise "Unmatched matries size: gxs.size(#{gxs.size}) != in_grad.size(#{in_grad.size})" unless gxs.size == in_grad.size

        unless retain_grad
          outputs.each do |y|
            unless y.nil? || y == @node
              grads[y] = nil
              y_var = y.get_variable
              y_var.grad_var = nil unless y_var.nil?
            end
          end
        end

        gxs.each_with_index do |gx, i|
          next if gx.nil?
          x = target_inputs[i]
          next unless x.requires_grad

          Utils::Variable.check_grad_type(func, x, gx.data)

          if target_inputs[0...i].include?(x)
            cur_gx = grads[x]
            grads[x] = cur_gx.nil? ? gx : gx + cur_gx
          else
            grads[x] = gx
          end

          x_var = x.get_variable
          x_var.grad_var = grads[x] if x_var

          if x.creator_node && !seen_set.include?(x.creator_node)
            funcs << x.creator_node
            seen_set.add(x.creator_node)
          end
        end

        funcs.sort_by! { |f| -f.rank }

      end
    end

    # Deletes the reference to the creator of this variable.
    #
    # This method deletes the reference to the creator from the corresponding
    # variable node. Unlike :meth:`unchain_backward`, it does not backtrack
    # the graph.
    #
    # This method is equivalent to ``self.creator = None``.
    def unchain
        self.creator = nil
    end

    # Deletes references between variable nodes and functions backward.
    #
    # After this method completes, intermediate variable nodes and functions
    # that are not referenced from anywhere are deallocated by reference
    # count GC. Also this variable itself deletes the reference to its
    # creator function from the node, i.e. the node becomes root in the
    # computation graph. It indicates that backprop after unchaining stops at
    # this variable. This behavior is useful to implement truncated BPTT.
    def unchain_backward
      cand_funcs = []
      seen_set = Set.new()

      add_cand = Proc.new do |cand|
        if cand && !seen_set.include?(cand)
          cand_funcs.append(cand)
          seen_set.add(cand)
        end
      end

      add_cand.(self.creator)

      while cand_funcs.size > 0
        func = cand_funcs.pop
        func.inputs.each do |var|
          add_cand.(var.creator)
        end
        func.unchain()
      end
    end

    def -@
      Functions::Math::Neg.new.apply([self]).first
    end

    def +(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Add.new.apply([self, other])[0]
      else
        Functions::Math::AddConstant.new(other).apply([self])[0]
      end
    end

    def -(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Sub.new.apply([self, other])[0]
      else
        Functions::Math::AddConstant.new(-other).apply([self])[0]
      end
    end

    def *(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Mul.new.apply([self, other])[0]
      else
        Functions::Math::MulConstant.new(other).apply([self])[0]
      end
    end

    def /(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Div.new.apply([self, other])[0]
      else
        Functions::Math::MulConstant.new(1 / other).apply([self])[0]
      end
    end

    def **(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::PowVarVar.new.apply([self, other])[0]
      else
        Functions::Math::PowVarConst.new(other).apply([self])[0]
      end
    end

    def retain_data
      @node.data = @data
    end

    # when left side is Numeric value and right side is Chainer::Value, call this method.
    def coerce(other)
      other = self.data.class.new.fill(other) if other.kind_of?(Numeric)
      [Chainer::Variable.new(other, requires_grad: false), self]
    end
  end
end

