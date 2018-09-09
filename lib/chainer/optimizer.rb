module Chainer
  class Optimizer
    attr_accessor :target

    def setup(link)
      @target = link
      @t = 0
      @epoch = 0
      
      @hooks = {}
    end

    def add_hook(hook, name: nil)
      if !hook.class.method_defined?(:call)
        raise TypeError, 'hook function is not callable'
      end

      name = hook.name if name.nil?
      if @hooks[name]
        raise TypeError, "hook #{name} already exists"
      end
      @hooks[name] = hook
    end

    def call_hooks
      @hooks.values.each do |hook|
        _call_hook(hook)
        reallocate_cleared_grads
      end
    end

    def _call_hook(hook)
      if hook.methods.include?(:call_for_each_param)
        @target.params do |param|
          hook.(param.update_rule, param)
        end
      else
        hook.(self)
      end
    end

    def serialize(serializer)
      @t = serializer.('t', @t)
      @epoch = serializer.('epoch', @epoch)
      
      @target.namedparams() do |(name, param)|
        if param.respond_to?(:update_rule)
          param.update_rule.serialize(serializer[name.to_s])
        end
      end
    end

    # Registers a hook function.
    #
    # Hook function is typically called right after the gradient computation,
    # though the timing depends on the optimization method.
    #
    # Args:
    #     hook (function): Hook function. If ``hook.call_for_each_param`` is
    #         true, this hook function is called for each parameter by
    #         passing the update rule and the parameter. Otherwise, this hook
    #         function is called only once each iteration by passing the
    #         optimizer.
    #     name (str): Name of the registration. If omitted, ``hook.name`` is
    #         used by default.
    def add_hook(hook, name: nil)
      raise TypeError.new("hook function is not callable") unless hook.respond_to?("call")
      raise RuntimeError.new("call `setup` method before `add_hook` method") unless @hooks

      name = hook.inspect unless name
      raise KeyError.new("hook #{name} already exists") if @hooks.key?(name)
      @hooks[name] = hook
    end

    # Removes the specified hook function.
    #
    # Args:
    #     name (str): Name of the hook function to be removed. The hook
    #         function registered with this name will be removed.
    def remove_hook(name)
      @hooks.delete(name)
    end
  end

  class UpdateRule
    attr_reader :state

    def initialize(parent_hyperparam:)
      @hooks = {}  
      @state = nil
      @enabled = true
      @hyperparam = Chainer::Hyperparameter.new(parent: parent_hyperparam)
      @t = 0
    end

    def update(param)
      return unless @enabled

      @t += 1
      unless param.data.nil?
        prepare(param)
      end
      @hooks.values.each do |hook|
        hook.call(param)
      end
      update_core(param)
    end

    def update_core(param)
      xm = Chainer.get_array_module(param)
      if xm == Cumo
        update_core_gpu(param)
      else
        update_core_cpu(param)
      end
    end

    def update_core_cpu
      raise NotImplementedError
    end

    def update_core_gpu
      raise NotImplementedError
    end

    def init_state(param)
      #raise NotImplementedError
    end


    # Serializes the update rule state.
    # Be careful that this method only saves/loads the state of the update rule.
    # The parameters of the target link is not saved/loaded by this
    # method, and so you need to serialize the target link separately if you
    # want to fully recover the training state including parameters.
    #
    # @param [Chainer::AbstractSerializer] serializer Serializer object.
    def serialize(serializer)
      if @state.nil?
        if serializer.is_a?(Chainer::Deserializer)
          # try to initialize the state to retrieve state entries
          @state = {}
          self_copy = self.dup
          # TODO(sonots): pass device from outside
          xm = Chainer::Device.default.xm
          arr = xm::SFloat.new(1)
          self_copy.init_state(Chainer::Variable.new(arr, grad: arr))
          @state.keys.each do |key|
            @state[key] = serializer.(key.to_s, nil)
          end
        end
      else
        @state.each do |key, val|
          @state[key] = serializer.(key.to_s, val)
        end
      end                                                                                 
    end

    private

    def prepare(param)
      if @state.nil?
        @state = {}
        init_state(param)
      end
      @state.select! { |_, v| Chainer.array?(v) }
    end
  end

  class HyperparameterProxy  
    def initialize(obj, attr_name)
      obj.class.class_eval do
        obj.class.send(:define_method, attr_name) do
          self.instance_variable_get(:@hyperparam).instance_variable_get("@#{attr_name}")
        end

        obj.class.send(:define_method, "#{attr_name}=") do |val|
          self.instance_variable_get(:@hyperparam).instance_variable_set("@#{attr_name}", val)
        end
      end
    end
  end

  # Optimizer/UpdateRule hook function for weight decay regularization
  #
  # This hook function adds a scaled parameter to the correspondeing gradient
  # It can be used as a regularization
  #
  # @param [Float] rate Coefficient for the weight decay
  class WeightDecay
    def name
      "WeightDecay"
    end

    def call_for_each_param
      true
    end

    def initialize(rate)
      @rate = rate
    end

    def call(rule, param)
      return if param.data.nil? || param.grad.nil?
      param.grad += (@rate * param.data)
    end
  end
end
