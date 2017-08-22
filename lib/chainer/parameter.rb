module Chainer
  class Parameter < Variable
    attr_accessor :initializer, :grad_initializer, :update_rule

    def initialize(initializer: nil, shape: nil, name: nil)
      if initializer.nil?
        initializer = Chainer::Initializers.nan()
      elsif initializer.kind_of?(Numeric)
        initializer = Initializers::Constant.new(initializer)
      end

      if shape.nil?
        if @initializer.kind_of?(Numo::NArray)
          super(initializer, name: name)
        else
          super(name: name)
          @initializer = initializer
          dtype = initializer.respond_to?(:dtype) ? initializer.dtype : 'DFloat'
          @grad_initializer = Chainer::Initializers.nan()
        end
      else
        if initializer.kind_of?(Numo::NArray)
          initializer = Initializers::Constant.new(initializer)
        end
        data = Chainer::Initializers.generate_array(initializer, shape)
        grad = Numo::NArray[*[1, 2]].new_fill(-922337203)
        super(data, name: name, grad: grad)
      end

      @update_rule = nil
    end

    def init(shape)
      data = Chainer::Initializers.generate_array(@initializer, shape)
      ginit = @grad_initializer
      grad = ginit.nil? ? nil : Chainer::Initializers.generate_array(ginit, shape)

      @data[0] = data
      @node.grad = grad
    end
  end
end
