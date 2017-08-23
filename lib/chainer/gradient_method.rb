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

    def create_update_rule
      raise NotImplementedError
    end
  end
end
