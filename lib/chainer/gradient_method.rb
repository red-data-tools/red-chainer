module Chainer
  class GradientMethod < Chainer::Optimizer
    def initialize
      super()
      @hyperparam = Hyperparameter.new
    end
    
    def setup(link)
      super(link)
      link.params do |param|
        param.update_rule = create_update_rule
      end
    end

    def reallocate_cleared_grads
      @target.namedparams(include_uninit: false) do |(name, param)|
        if param.grad.nil?
          xm = Chainer.get_array_module(param.data)
          param.grad = xm::NArray.[](*param.data).new_zeros
        end
      end
    end

    def call_hooks
      @hooks.values.each do |hook|
        _call_hook(hook)
        reallocate_cleared_grads
      end
    end

    def update(lossfun=nil, *args, **kwds)
      if lossfun
        use_cleargrads = self.methods.include?(:use_cleargrads) ? self.use_cleargrads : true
        if args.size > 0 && kwds.keys.size > 0
          loss = lossfun.(*args, **kwds)
        elsif args.size > 0
          loss = lossfun.(*args)
        elsif kwds.keys.size > 0
          loss = lossfun.(**kwds)
        end

        if use_cleargrads
          @target.cleargrads()
        else
          @target.zerograds()
        end
        loss.backward()
      end

      reallocate_cleared_grads

      call_hooks

      @t += 1
      @target.params do |param|
        param.update
      end
    end

    def create_update_rule
      raise NotImplementedError
    end
  end
end
