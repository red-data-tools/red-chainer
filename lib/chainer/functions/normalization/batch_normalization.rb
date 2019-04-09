module Chainer
  module Functions
    module Normalization
      module Calculation
        def apply_bn_fwd(xp, x, mean, inv_std, gamma, beta)
          # NOTE: all arguments should be broadcasted to x.shape
          # (mean, inv_std, gamma, and beta have to already be expanded)
          x_hat = x_hat(x, mean, inv_std)
          y = gamma * x_hat
          y += beta
          y
        end

        def x_hat(x, mean, inv_std)
          x_mu = x - mean
          x_mu *= inv_std
          x_mu
        end

        def zero_if_none(xp, x, shape, dtype)
          # TODO: Return broadcasted 0 instead of a zeroed array.
          x.nil? ? dtype.zeros(*shape) : x
        end

        def compute_axis(x_ndim, gamma_ndim=1, axis=nil)
          if axis.nil?
            # [0, 2, 3]
            axis = [0] + ((gamma_ndim + 1) ...  x_ndim).to_a
          end
          axis
        end

        def compute_key_axis(x_ndim, gamma_ndim=1, axis=nil)
          axis = compute_axis(x_ndim, gamma_ndim, axis)
          # [1]
          (0...x_ndim).to_a - axis
        end

        def reduced_shape(x_shape, axis, keepdims: false)
          if keepdims
            reduced_shape = x_shape.dup
            axis.each {|i| reduced_shape[i] = 1 }
          else
            reduced_shape = x_shape.dup
            axis.reverse.each {|i| reduced_shape.delete_at(i) }
          end
          reduced_shape
        end

        def can_use_cudnn?(axis)
          # cuDNN restriction
          return true if (
            axis.size == 3 and
            axis[0] == 0 and axis[1] == 2 and axis[2] == 3
          )
          return true if (
            axis.size == 4 and
            axis[0] == 0 and axis[1] == 2 and axis[2] == 3 and axis[3] == 4
          )
          false
        end
      end

      class BatchNormalization < Chainer::FunctionNode
        include Calculation
        attr_reader :running_mean, :running_var

        def self.batch_normalization(x, gamma, beta, eps: 2e-5, running_mean: nil, running_var: nil, decay: 0.9)
          BatchNormalization.new(eps: eps, mean: running_mean, var: running_var, decay: decay).apply([x, gamma, beta])[0]
        end

        def initialize(eps: 2e-5, mean: nil, var: nil, decay: 0.9)
          @mean = nil
          @inv_std = nil

          @running_mean = mean
          @running_var = var
          @eps = eps
          # TODO: raise error if eps < CUDNN_BN_MIN_EPSILON
          @decay = decay
          @use_cudnn = false
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x, gamma, beta = inputs

          @axis = compute_axis(x.ndim, gamma.ndim)
          @key_axis = compute_key_axis(x.ndim, gamma.ndim)

          # expander inserts singleton dimensions to gamma and beta so that they
          # can be broadcasted with x.
          head_ndim = gamma.ndim + 1
          # TODO: expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
          suffix = [1] * (x.ndim - head_ndim)
          expander = -> (arr) do
            shape = [1] + arr.shape + suffix
            arr.reshape(*shape)
          end
          @expander = expander

          xp = Chainer.get_array_module(x)
          @use_cudnn = (xp == Cumo and can_use_cudnn?(@axis))

          if @use_cudnn
            y = _forward_cudnn(x, gamma, beta)
          else
            gamma = expander.(gamma)
            beta = expander.(beta)
            @mean = x.mean(axis: @axis)

            # TODO: Numo::Array can not be specified standard deviation
            var = ((x - x.mean(axis: @axis, keepdims: true)) ** 2).mean(axis: @axis)

            var += @eps
            @inv_std = var ** (-0.5)

            y = apply_bn_fwd(xp, x, expander.(@mean), expander.(@inv_std), gamma, beta)
            # Update running statistics
            m = x.size.div(gamma.size)
            adjust = m / [m - 1.0, 1.0].max
            if !@running_mean.nil?
              @running_mean *= @decay
              @running_mean += (1 - @decay) * @mean
              @running_var *= @decay
              @running_var += (1 - @decay) * adjust * var
            end
          end

          [y]
        end

        private def _forward_cudnn(x, gamma, beta)
          # batch_norm allocates NArray memory, not need calling #zeros
          if @running_mean.nil? # create dummies
            @running_mean = x.class.new(*gamma.shape)
            @running_var = x.class.new(*gamma.shape)
          end
          @mean = x.class.new(*gamma.shape)
          @inv_std = x.class.new(*gamma.shape)
          y = x.batch_norm(
            gamma,
            beta,
            running_mean: @running_mean,
            running_var: @running_var,
            mean: @mean,
            inv_std: @inv_std,
            eps: @eps,
            decay: @decay,
            axis: @axis)
          y
        end

        def backward(indexes, grad_outputs)
          x, gamma = get_retained_inputs
          gy, = grad_outputs

          # hatappi debug
          #@mean = @mean.class.new(@mean.shape).seq
          #@inv_std = @inv_std.class.new(@inv_std.shape).seq
          #x.data = x.data.class.new(x.shape).seq
          #gamma.data = gamma.data.class.new(gamma.shape).seq
          #gy.data = gy.data.class.new(gy.shape).seq

          f = BatchNormalizationGrad.new(@eps, @use_cudnn, @expander, @axis, @mean, @inv_std)
          f.(x, gamma, gy)
        end
      end

      class BatchNormalizationGrad < Function
        include Calculation

        def initialize(eps, use_cudnn, expander, axis, mean, inv_std)
          @eps = eps
          @use_cudnn = use_cudnn
          @expander = expander
          @axis = axis
          @mean = mean
          @inv_std = inv_std
        end

        def forward(inputs)
          retain_inputs([0, 1, 2])
          x, gamma, gy = inputs

          if @use_cudnn
            gx, ggamma, gbeta = _forward_cudnn(x, gamma, gy)
          else
            expander = @expander

            inv_m = gamma.class.new.fill(1.0 / x.size.div(gamma.size))
            xp = Chainer.get_array_module(x)

            gbeta = gy.sum(axis: @axis)
            x_hat = x_hat(x, expander.(@mean), expander.(@inv_std))
            ggamma = (gy * x_hat).sum(axis: @axis)
            gx = expander.(gamma * @inv_std) * (gy - (x_hat * expander.(ggamma) + expander.(gbeta)) * inv_m)
          end

          retain_outputs([0, 1])
          [gx, ggamma, gbeta]
        end

        private def _forward_cudnn(x, gamma, gy)
          return x.batch_norm_backward(
            gamma,
            gy,
            mean: @mean,
            inv_std: @inv_std,
            eps: @eps,
            axis: @axis)
        end

        def backward(inputs, grad_outputs)
          expander = @expander

          x, gamma, gy = inputs
          gx1, ggamma1, = output_data
          ggx1, gggamma1, ggbeta1 = grad_outputs
          xp = Chainer.get_array_module(x)

          # auxiliary values
          inv_m = gamma.class.new.fill(1.0 / x.size.div(gamma.size))
          r = ggx1.nil? ? 0 : (gx1 * ggx1).sum(axis: @axis)
          coeff = gamma * @inv_std
          coeff_m = coeff * inv_m
          x_hat = x_hat(x, expander.(@mean), expander.(@inv_std))

          # handle None in output gradients
          ggx1 = zero_if_none(xp, ggx1, x.shape, x.class)
          gggamma1 = zero_if_none(xp, gggamma1, gamma.shape, gamma.class)
          ggbeta1 = zero_if_none(xp, ggbeta1, gamma.shape, gamma.class)

          gggamma2 = gggamma1 - coeff_m * (x_hat * ggx1).sum(axis: @axis)
        	ggbeta2 = ggbeta1 - coeff_m * ggx1.sum(axis: @axis)

          ggamma2 = r / gamma

          gx_hat2 = (expander.(gggamma2) * gy - expander.(coeff_m * ggamma1) * ggx1)
          gstd2 = -@inv_std * (r + (x_hat * gx_hat2).sum(axis: @axis))
          gmean2 = -@inv_std * gx_hat2.sum(axis: @axis)
          gx2 = expander.(@inv_std) * gx_hat2 + inv_m * (expander.(gmean2) + x_hat * expander.(gstd2))
          ggy2 = (expander.(gggamma2) * x_hat + expander.(ggbeta2) + expander.(coeff) * ggx1)

          [gx2, ggamma2, ggy2]
        end
      end

      class FixedBatchNormalization < FunctionNode
        include Calculation

        attr_reader :inv_var

        def self.fixed_batch_normalization(x, gamma, beta, mean, var, eps: 2e-5)
          FixedBatchNormalization.new(eps: eps).apply([x, gamma, beta, mean, var]).first
        end

        def initialize(eps: 2e-5)
          @inv_std = nil
          @inv_var = nil
          @eps = eps
        end

        def forward(inputs)
          retain_inputs([0, 1, 3, 4])
          x, gamma, beta, mean, var = inputs
          xp = Chainer.get_array_module(x)

          # expander inserts singleton dimensions to gamma and beta so that they
          # can be broadcasted with x.
          head_ndim = gamma.ndim + 1
          # TODO: expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
          suffix = [1] * (x.ndim - head_ndim)
          expander = -> (arr) do
            shape = [1] + arr.shape + suffix
            arr.reshape(*shape)
          end
          @expander = expander
          @axis = [0] + (head_ndim...(x.ndim)).to_a

          gamma = expander.(gamma)
          beta = expander.(beta)
          var += @eps
          @inv_var = var.reciprocal
          @inv_std = xp::NMath.sqrt(@inv_var)

          y = apply_bn_fwd(xp, x, expander.(mean), expander.(@inv_std), gamma, beta)
          [y]
        end

        def backward(indexes, grad_outputs)
          x, gamma, mean, var = get_retained_inputs
          gy, = grad_outputs
          f = FixedBatchNormalizationGrad.new(@eps, @expander, @axis, @inv_std, @inv_var)
          f.(x, gamma, mean, var, gy)
        end
      end

      class FixedBatchNormalizationGrad < Function
        include Calculation

        def initialize(eps, expander, axis, inv_std, inv_var)
          @eps = eps
          @expander = expander
          @axis = axis
          @inv_std = inv_std
          @inv_var = inv_var
        end

        def forward(inputs)
          retain_inputs([0, 1, 2, 4])
          x, gamma, mean, var, gy = inputs
          expander = @expander
          xp = Chainer.get_array_module(x)

          if @inv_std.nil? || @inv_var.nil?
            @inv_var = (var + @eps).reciprocal
            @inv_std = xp::NMath.sqrt(@inv_var)
          end

          @gamma_over_std = gamma * @inv_std
          x_hat = x_hat(x, expander.(mean), expander.(@inv_std))

          gx = expander.(@gamma_over_std) * gy
          gbeta = gy.sum(axis: @axis)
          ggamma = (x_hat * gy).sum(axis: @axis)
          gmean = -@gamma_over_std * gbeta
          gvar = -0.5 * gamma * @inv_var * ggamma

          retain_outputs([0, 1, 2, 3, 4])
          [gx, ggamma, gbeta, gmean, gvar]
        end

        def backward(inputs, grad_outputs)
          x, gamma, mean, _, gy = inputs
          ggx1, gggamma1, ggbeta1, ggmean1, ggvar1 = grad_outputs
          gx1, ggamma1, gbeta1, gmean1, gvar1 = output_data

          # Handle None in output gradients.
          xp = Chainer.get_array_module(x)
          ggx1 = zero_if_none(xp, ggx1, x.shape, x.class)
          gggamma1 = zero_if_none(xp, gggamma1, gamma.shape, gamma.class)
          ggbeta1 = zero_if_none(xp, ggbeta1, gamma.shape, gamma.class)
          ggmean1 = zero_if_none(xp, ggmean1, mean.shape, mean.class)
          ggvar1 = zero_if_none(xp, ggvar1, mean.shape, mean.class)

          expander = @expander
          x_hat = x_hat(x, expander.(mean), expander.(@inv_std))
          tmp = -0.5 * ggvar1

          gamma_over_var = gamma * @inv_var
          g_gamma_over_var = tmp * ggamma1

          gggamma2 = gggamma1 + tmp * gamma_over_var
          gx_hat = gy * expander.(gggamma2)
          gx2 = expander.(@inv_std) * gx_hat
          gmean2 = -@inv_std * gx_hat.sum(axis: @axis)

          g_gamma_over_std = (ggx1 * gy).sum(axis: @axis) - ggmean1 * gbeta1
          ggbeta2 = ggbeta1 - ggmean1 * @gamma_over_std
          ggy2 = (expander.(gggamma2) * x_hat + expander.(ggbeta2) + expander.(@gamma_over_std) * ggx1)

          ggamma2 = (@inv_var * g_gamma_over_var + @inv_std * g_gamma_over_std)
          gvar2 = -(ggamma2 * gamma_over_var + 0.5 * @inv_var * ((x_hat * gx_hat).sum(axis: @axis) - @gamma_over_std * g_gamma_over_std))

          [gx2, ggamma2, gmean2, gvar2, ggy2]
        end
      end
    end
  end
end
