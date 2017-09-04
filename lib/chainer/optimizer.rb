module Chainer
  class Optimizer
    attr_accessor :target

    def setup(link)
      @target = link
      @t = 0
      @epoch = 0
      
      @hooks = {}
    end

    def _call_hook(hook)
      if hook.methods.include?(:call_for_each_param)
        @target.params.each do |param|
          hook.(param.update_rule, param)
        end
      else
        hook(self)
      end
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
      prepare(param)
      @hooks.values.each do |hook|
        hook.call(param)
      end
      update_core(param)
    end

    def update_core(param)
      # TODO: support GPU
      update_core_cpu(param)
    end

    def update_core_cpu
      raise NotImplementedError
    end

    def init_state(param)
      raise NotImplementedError
    end

    private

    def prepare(param)
      if @state.nil?
        @state = {}
        init_state(param)
      end
      @state.select! { |_, v| v.kind_of?(Numo::NArray) }
    end
  end
end
