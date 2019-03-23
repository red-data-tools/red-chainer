module Chainer
  module Links
    module Normalization 
      class BatchNormalization < Chainer::Link
        # Batch normalization layer on outputs of linear or convolution functions.
        # 
        # It runs in three modes: training mode, fine-tuning mode, and testing mode.
        # In training mode, it normalizes the input by *batch statistics*. It also
        # maintains approximated population statistics by moving averages, which can
        # be used for instant evaluation in testing mode.
        # 
        # In fine-tuning mode, it accumulates the input to compute *population
        # statistics*. In order to correctly compute the population statistics, a
        # user must use this mode to feed mini-batches running through whole training dataset.
        # 
        # In testing mode, it uses pre-computed population statistics to normalize the input variable.
        # The population statistics is approximated if it is computed by training mode,
        # or accurate if it is correctly computed by fine-tuning mode.
        #
        # @param [integer or int array] size Size (or shape) of channel dimensions.
        # @param [float] decay Decay rate of moving average. It is used on training.
        # @param [float] eps Epsilon value for numerical stability.
        # @param [Numo::NArray.dtype or Cumo::NArray.dtype] dtype Type to use in computing.
        # @param [boolean] use_gamma If `true`, use scaling parameter. Otherwise, use unit(1) which makes no effect.
        # @param [boolean] use_beta If `true`, use shifting parameter. Otherwise, use unit(0) which makes no effect.
        def initialize(size, decay: 0.9, eps: 2e-5, dtype: nil, use_gamma: true, use_beta: true, initial_gamma: nil, initial_beta: nil)
          super()
          dtype ||= Chainer::Device.default.xm::SFloat
          @avg_mean = dtype.zeros(size)
          register_persistent('avg_mean')
          @avg_var = dtype.zeros(size)
          register_persistent('avg_var')
          @n = 0
          register_persistent('n')
          @decay = decay
          @eps = eps

          init_scope do
            if use_gamma
              initial_gamma = 1 if initial_gamma.nil?
              initial_gamma = Chainer::Initializers.get_initializer(initial_gamma)
              initial_gamma.dtype = dtype
              @gamma = Chainer::Parameter.new(initializer: initial_gamma, shape: size)
            end
            if use_beta
              initial_beta = 0 if initial_beta.nil?
              initial_beta = Chainer::Initializers.get_initializer(initial_beta)
              initial_beta.dtype = dtype
              @beta = Chainer::Parameter.new(initializer: initial_beta, shape: size)
            end
          end
        end

        # Invokes the forward propagation of BatchNormalization.
        # In training mode, the BatchNormalization computes moving averages of
        # mean and variance for evaluatino during training, and normalizes the input using batch statistics.
        # @param [Chainer::Variable] x Input variable.
        # @param [boolean] finetune If it is in the training mode and `finetune` is `True`,
        # BatchNormalization runs in fine-tuning mode;
        # it accumulates the input array to compute population statistics for normalization,
        # and normalizes the input using batch statistics.
        def call(x, finetune: false)
          if self.instance_variable_defined?(:@gamma)
            gamma = @gamma
          else
            gamma = Chainer::Variable.new(x.data.class.ones(@avg_mean.shape))
          end

          if self.instance_variable_defined?(:@beta)
            beta = @beta
          else
            beta = Chainer::Variable.new(x.data.class.zeros(*@avg_mean.shape))
          end
          
          if Chainer.configuration.train
            if finetune
              @n += 1
              decay = 1.0 - 1.0 / @n
            else
              decay = @decay
            end

            ret = Chainer::Functions::Normalization::BatchNormalization.batch_normalization(x, gamma, beta, eps: @eps, running_mean: @avg_mean, running_var: @avg_var, decay: decay)
          else
            mean = Chainer::Variable(@avg_mean)
            var = Chainer::Variable(@avg_var)
            ret = Chainer::Functions::Normalization::FixedBatchNormalization.fixed_batch_normalization(x, gamma, beta, mean, var, eps: @eps)
          end

          ret
        end

        # Resets the population count for collecting population statistics.
        # This method can be skipped if it is the first time to use the fine-tuning mode.
        # Otherwise, this method should be called before starting the fine-tuning mode again.
        def start_finetuning
          @n = 0
        end
      end
    end
  end
end

