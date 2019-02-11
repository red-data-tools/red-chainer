module Chainer
  class Variable
    attr_accessor :data, :grad, :requires_grad, :node

    def initialize(data=nil, name: nil, grad: nil, requires_grad: true)
      unless data.nil? || Chainer.array?(data)
        raise TypeError, "Numo::NArray or Cumo::NArray are expected."
      end

      @data = [data]
      @grad = grad
      @requires_grad = requires_grad
      @node = VariableNode.new(variable: self, name: name)
      @grad_var = grad.nil? ? nil : Chainer::Variable.new(grad)
    end

    def data
      return @data[0]
    end

    def data=(d)
      @data[0] = d
      @node.set_data_type(d)
    end

    def name
      return @node.name
    end

    def name=(n)
      @node.name = n
    end

    def label
      @node.label
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
      @grad_var = g.nil? ? nil : Chainer::Variable.new(g)
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

    def reshape(*shape)
      self.data.reshape(*shape)
    end

    # Clears the gradient array.
    def cleargrad
      @grad_var = nil
    end

    # Notifies the variable that the given node is its creator.
    #
    # @param [Chainer::FunctionNode] Function node that has this variable as an output.
    def set_creator_node(fnode)
      @node.set_creator_node(fnode)
    end

    def backward(retain_grad: false)
      return if self.creator_node.nil?

      grads = {}
      if self.data.size == 1 && self.grad_var.nil?
        self.grad = self.data.new_ones
      end
      grads[self.node] = self.grad_var

      funcs = [self.creator_node]

      while func = funcs.pop
        inputs = func.inputs
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
        # When the same variable is passed to multiple input slots (e.g. an expression like `f(x, x)`),
        # it makes the gradient accumulation complicated since the back-propagated gradients w.r.t.
        # the first and second argument should be accumulated to the current gradient w.r.t. the same variable.
        # In this case, the current implementation passes the current gradient only to the first occurrence of the variable
        # in the input tuple and passes `nil` to the rest of the occurrences.
        # For example, when the input variables are `(x, x)`,
        # the input gradient passed to the `backward_accumulate` method is `(gx, nil)` where `gx` is the current gradient of ``x``.
        # See also the docstring of `FunctionNode.backward_accumulate`.
        inputs.each_with_index.map { |x, i| i if x.requires_grad }
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
              y.grad = nil
            end
          end
        end

        seen_vars = []
        need_copy = []

        func.inputs.zip(gxs).each do |x, gx|
          next if gx.nil?
          next unless x.requires_grad

          Utils::Variable.check_grad_type(func, x, gx)

          id_x = x.object_id
          if x.creator.nil? # leaf
            if x.grad.nil?
              x.grad = gx
              need_copy << id_x
            else
              if need_copy.include?(id_x)
                x.grad = Utils::Array.force_array(x.grad + gx)
                need_copy.delete(id_x)
              else
                x.grad += gx
              end
            end
          else # not leaf
            funcs << x.creator
            if seen_vars.include?(id_x)
              if need_copy.include?(id_x)
                x.grad = Utils::Array.force_array(gx + x.grad)
                need_copy.delete(id_x)
              else
                x.grad += gx
              end
            else
              x.grad = gx
              seen_vars << id_x
              need_copy << id_x
            end
          end
        end
      end 
    end

    def -@
      Functions::Math::Neg.new.(self) 
    end

    def +(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Add.new.(*[self, other])
      else
        Functions::Math::AddConstant.new(other).(self)
      end
    end

    def -(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Sub.new.apply(*[self, other])[0]
      else
        Functions::Math::AddConstant.new(-other).apply([self])[0]
      end
    end

    def *(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Mul.new.apply(*[self, other])[0]
      else
        Functions::Math::MulConstant.new(other).apply([self])[0]
      end
    end

    def /(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Div.new.apply(*[self, other])[0]
      else
        Functions::Math::MulConstant.new(1 / other).apply([self])[0]
      end
    end

    def **(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::PowVarVar.new.apply(*[self, other])[0]
      else
        Functions::Math::PowVarConst.new(other).apply([self])[0]
      end
    end

    def retain_data
      @node.data = @data[0]
    end

    # when left side is Numeric value and right side is Chainer::Value, call this method.
    def coerce(other)
      other = self.data.class[*other] if other.kind_of?(Numeric)
      [Chainer::Variable.new(other, requires_grad: false), self]
    end
  end
end

