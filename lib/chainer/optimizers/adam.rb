module Chainer
  module Optimizers
    class AdamRule < UpdateRule
      def initialize(parent_hyperparam: nil, alpha: nil, beta1: nil, beta2: nil, eps: nil)
        hyperparam = Hyperparameter.new
        hyperparam.instance_variable_set('@alpha', 0.001)
        hyperparam.instance_variable_set('@beta1', 0.9)
        hyperparam.instance_variable_set('@beta2', 0.999)
        hyperparam.instance_variable_set('@eps', 1e-8)

        super(parent_hyperparam: parent_hyperparam || hyperparam)

        @hyperparam.instance_variable_set('@alpha', alpha) if alpha
        @hyperparam.instance_variable_set('@beta1', beta1) if beta1
        @hyperparam.instance_variable_set('@beta2', beta2) if beta2
        @hyperparam.instance_variable_set('@eps', eps) if eps
      end

      def init_state(param)
        @state[:m] = param.data.new_zeros
        @state[:v] = param.data.new_zeros
      end

      def update_core(param)
        grad = param.grad
        return if grad.nil?

        hp = @hyperparam

        @state[:m] += (1 - hp.beta1) * (grad - @state[:m])
        @state[:v] += (1 - hp.beta2) * (grad * grad - @state[:v])
        xm = Chainer.get_array_module(grad)
        param.data -= lr * @state[:m] / (xm::NMath.sqrt(@state[:v]) + hp.eps)
      end

      def lr
        fix1 = 1.0 - @hyperparam.beta1 ** @t
        fix2 = 1.0 - @hyperparam.beta2 ** @t
        @hyperparam.alpha * Math.sqrt(fix2) / fix1
      end
    end

    class Adam < GradientMethod
      def initialize(alpha: nil, beta1: nil, beta2: nil, eps: nil)
        super()
        @hyperparam.instance_variable_set('@alpha', alpha || 0.001)
        @hyperparam.instance_variable_set('@beta1', beta1 || 0.9)
        @hyperparam.instance_variable_set('@beta2', beta2 || 0.999)
        @hyperparam.instance_variable_set('@eps', eps || 1e-8)
      end

      def create_update_rule
        AdamRule.new(parent_hyperparam: @hyperparam)
      end

      def lr
        fix1 = 1.0 - (@hyperparam.beta1 ** @t)
        fix2 = 1.0 - (@hyperparam.beta2 ** @t)
        @hyperparam.alpha * Math.sqrt(fix2) / fix1
      end
    end
  end
end
