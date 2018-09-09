module Chainer
  module Optimizers
    # Update rule of vanilla stochastic gradient descent.

    # See :class:`~chainer.optimizers.SGD` for the default values of the
    # hyperparameters.

    # Args:
    #     parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
    #         that provides the default values.
    #     lr (float): Learning rate.
    class SGDRule < UpdateRule

      def initialize(parent_hyperparam: nil, lr: nil)
        if parent_hyperparam
          super(parent_hyperparam: parent_hyperparam)
        else
          hyperparam = Hyperparameter.new
          hyperparam.instance_variable_set("@lr", 0.01)
          super(parent_hyperparam: hyperparam)
        end

        @hyperparam.instance_variable_set("@lr", lr) if lr
      end

      def update_core_cpu(param)
        grad = param.grad
        return unless grad
        param.data -= @hyperparam.lr * grad
      end
    end


    # Vanilla Stochastic Gradient Descent.
    # Args:
    #     lr (float): Learning rate.
    class SGD < GradientMethod

      def initialize(lr: 0.01)
        super()
        @hyperparam.instance_variable_set("@lr", lr)
        Chainer::HyperparameterProxy.new(self, "lr")
      end

      def create_update_rule
        SGDRule.new(parent_hyperparam: @hyperparam)
      end
    end
  end
end
