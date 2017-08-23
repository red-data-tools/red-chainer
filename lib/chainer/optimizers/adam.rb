module Chainer
  module Optimizers
    class Adam < GradientMethod
      def initialize(alpha: nil, beta1: nil, beta2: nil, eps: nil)
        super()
        @hyperparam.instance_variable_set('@alpha', alpha || 0.001)
        @hyperparam.instance_variable_set('@beta1', alpha || 0.9) 
        @hyperparam.instance_variable_set('@beta2', alpha || 0.999) 
        @hyperparam.instance_variable_set('@eps', alpha || 1e-8)
      end

      def lr
        fix1 = 1.0 - (@hyperparam.beta1 ** @t)
        fix2 = 1.0 - (@hyperparam.beta2 ** @t)
        @hyperparam.alpha * Math.sqrt(fix2) / fix1
      end
    end
  end
end
