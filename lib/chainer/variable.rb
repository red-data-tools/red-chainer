module Chainer
  class Variable
    attr_accessor :data, :grad, :requires_grad, :node

    def initialize(data=nil, **kwargs)
      args = Utils::Argument.parse_kwargs(kwargs, name: nil, grad: nil, requires_grad: true)
      unless data.nil? || data.is_a?(Numo::NArray)
        raise TypeError, "Numo::NArray are expected."
      end

      @data = [data]
      @grad = args[:grad]
      @requires_grad = args[:requires_grad]
      @node = VariableNode.new(variable: self, name: args[:name], grad: args[:grad])
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

    def creator
      @node.creator
    end

    def creator=(func)
      @node.creator = func
    end

    def grad
      @node.grad
    end

    def grad=(g)
      @node.set_grad_with_check(g, nil, self)
    end

    def shape
      self.data.shape
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

    def cleargrad
      @node.grad = nil
    end

    def set_creator(gen_func)
      @node.set_creator(gen_func)
    end

    def backward(retain_grad: false)
      return if self.creator.nil?

      if self.data.size == 1 && self.grad.nil?
        self.grad = self.data.new_ones
      end

      funcs = [self.creator]

      while func = funcs.pop
        outputs = func.outputs.map(&:__getobj__)
        in_data = func.inputs.map(&:data)
        out_grad = outputs.map { |y| y.nil? ? nil : y.grad }

        func.output_data = outputs.map { |y| y.nil? ? nil : y.data }
        gxs = func.backward(in_data, out_grad)

        raise unless gxs.size == in_data.size

        unless func.retain_after_backward
          func.output_data = nil
        end

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
        Functions::Math::Sub.new.(*[self, other])
      else
        Functions::Math::AddConstant.new(-other).(self)
      end
    end

    def *(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Mul.new.(*[self, other])
      else
        Functions::Math::MulConstant.new(other).(self)
      end
    end

    def /(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Div.new.(*[self, other])
      else
        Functions::Math::MulConstant.new(1 / other).(self)
      end
    end

    def **(other) 
      if other.instance_of?(Chainer::Variable)
        Functions::Math::PowVarVar.new.(*[self, other])
      else
        Functions::Math::PowVarConst.new(other).(self)
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

