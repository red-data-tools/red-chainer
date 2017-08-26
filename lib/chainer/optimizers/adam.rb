module Chainer
  module Optimizers

    class Adam < GradientMethod
      def initialize(alpha: nil, beta1: nil, beta2: nil, eps: nil)
        super()
        @hyperparam.instance_variable_set('@alpha', alpha || 0.001)
        @hyperparam.instance_variable_set('@beta1', beta1 || 0.9)
        @hyperparam.instance_variable_set('@beta2', beta2 || 0.999)
        @hyperparam.instance_variable_set('@eps', eps || 1e-8)
      end

      def create_update_rule
        # TODO
      end

      def lr
        fix1 = 1.0 - (@hyperparam.beta1 ** @t)
        fix2 = 1.0 - (@hyperparam.beta2 ** @t)
        @hyperparam.alpha * Math.sqrt(fix2) / fix1
      end
    end
  end
end
