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
end
