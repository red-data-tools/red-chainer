module Chainer
  module Functions
    module Normalization
      class BatchNormalizationFunction < Chainer::Function
        attr_reader :running_mean, :running_var
        # Batch normalization function with fixed statistics.
        # This is a variant of batch normalization, where the mean and variance
        # statistics are given by the caller as fixed variables. This is
        # used on testing mode of the batch normalization layer, where batch
        # statistics cannot be used for prediction consistency.
        #
        # @param [Chainer::Variable] x Input variable.
        # @param [Chainer::Variable] gamma Scaling parameter of normalized data.
        # @param [Chainer::Variable] beta Shifting parameter of scaled normalized data.
        # @param [Chainer::Variable] mean Shifting parameter of input.
        # @param [Chainer::Variable] var Square of scaling parameter of input.
        # @param [float] eps Epsilon value for numerical stability.
        def self.fixed_batch_normalization(x, gamma, beta, mean, var, eps: 2e-5)
          old_train = Chainer.configuration.train
          Chainer.configuration.train = false
          norm = self.new(eps: eps, mean: nil, var: nil, decay: 0.0).(x, gamma, beta, mean, var)
          Chainer.configuration.train = old_train
          norm
        end
      
        def initialize(eps: 2e-5, mean: nil, var: nil, decay: 0.9) 
          @running_mean = mean
          @running_var = var
          @eps = eps
          @mean_cache = nil
          @decay = decay
        end

        def forward(inputs)
          x, gamma, beta = inputs[0], inputs[1], inputs[2]
          if Chainer.configuration.train
            if @running_mean.nil?
              @running_mean = Numo::NArray[*gamma].new_zeros
              @running_var = Numo::NArray[*gamma].new_zeros
            else
              @running_mean = Numo::NArray[*@running_mean]
              @running_var = Numo::NArray[*@running_var]
            end
          elsif inputs.size == 5
            @fixed_mean = inputs[3]
            @fixed_var = inputs[4]
          end

          head_ndim = gamma.ndim + 1
          gamma_expander = [1] + gamma.shape + [1] * (x.ndim - head_ndim)
          gamma = gamma.reshape(*gamma_expander)
          beta_expander = [1] + beta.shape + [1] * (x.ndim - head_ndim)
          beta = beta.reshape(*beta_expander)
        
          if Chainer.configuration.train
            axis = [0] + (head_ndim...(x.ndim)).to_a
            mean = x.mean(axis: axis)
            # FIXME: numpy.var
            var = x.var(axis: axis)
            var += @eps
          else
            mean = @fixed_mean
            var = @fixed_var + @eps
          end

          @std = Numo::NMath.sqrt(var)

          mean_expander = [1] + mean.shape + [1] * (x.ndim - head_ndim)
          x_mu = x - mean.reshape(*mean_expander)
          std_expander = [1] + @std.shape + [1] * (x.ndim - head_ndim)
          x_mu /= @std.reshape(*std_expander)
          @x_hat = x_mu
          y = gamma * @x_hat
          y += beta

          if Chainer.configuration.train
            m = x.size.div(gamma.size)
            adjust = m / [m - 1.0, 1.0].max
            @running_mean *= @decay
            temp_ar = Numo::NArray[*mean]
            temp_ar *= (1 - @decay)
            @running_mean += temp_ar
            
            @running_var *= @decay
            temp_ar = Numo::NArray[*var]
            temp_ar *= ((1 - @decay) * adjust)
            @running_var += temp_ar
          end

          [y,]
        end
      end
    end
  end
end
