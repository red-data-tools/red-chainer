module Chainer
  module Optimizers
    # Update rule for the classical momentum SGD
    class MomentumSGDRule < UpdateRule
      def initialize(parent_hyperparam: nil, lr: nil, mementum: nil)
        hyperparam = Hyperparameter.new
        hyperparam.instance_variable_set('@lr', 0.01)
        hyperparam.instance_variable_set('@momentum', 0.9)

        super(parent_hyperparam: parent_hyperparam || hyperparam)
        
        @hyperparam.instance_variable_set('@lr', lr) if lr
        @hyperparam.instance_variable_set('@mementum', mementum) if mementum
      end
   
      def init_state(param)
        @state[:v] = param.data.new_zeros
      end

      def update_core(param)
        grad = param.grad
        return if grad.nil?
          
        v = @state[:v]
        v *= @hyperparam.momentum
        v -= @hyperparam.lr * grad
        param.data += v
      end 
    end
    
    # Momentum SGD optimizer
    class MomentumSGD < GradientMethod
      attr_accessor :lr, :momentum
      # @param [Float] lr Learning rate
      # @param [Float] momentum Exponential decay rate of the first order moment
      def initialize(lr: nil, momentum: nil)
        super()
        @hyperparam.instance_variable_set('@lr', lr || 0.01)
        @hyperparam.instance_variable_set('@momentum', momentum || 0.9)
        Chainer::HyperparameterProxy.new(self, "lr")
        Chainer::HyperparameterProxy.new(self, "momentum")
      end
     
      def create_update_rule
        MomentumSGDRule.new(parent_hyperparam: @hyperparam)
      end
    end
  end
end
