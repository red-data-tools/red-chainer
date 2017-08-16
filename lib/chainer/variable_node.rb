module Chainer
  class VariableNode
    attr_reader :dtype, :shape
    attr_accessor :data, :name, :grad, :rank, :creator, :requires_grad, :variable

    def initialize(variable: , name:, grad: nil)
      @variable = WeakRef.new(variable)
      @creator = nil
      @data = nil
      @rank = 0
      @name = name
      @requires_grad = variable.requires_grad

      set_data_type(variable.data)

      @grad = grad
    end

    def creator=(func)
      @creator = func
      unless func.nil?
        @rank = func.rank + 1
      end
    end

    def data=(data)
      @data = data
      set_data_type(data)
    end

    def grad=(g)
      Utils::Variable.check_grad_type(nil, self, g)
      @grad = g
    end

    def label
      if @shape.nil? || @shape.empty?
        @dtype.to_s
      else
        "(#{@shape.join(', ')}), #{@dtype.to_s}"
      end
    end

    def set_creator(creator)
      self.creator = creator
    end

    def unchain
      @creator = nil
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
        @dtype = data.class.to_s.split('::').last
        @shape = data.shape
      end
    end

    def set_grad_with_check(g, func, var)
      Utils::Variable.check_grad_type(func, var, g)
      @grad = g
    end
  end
end
