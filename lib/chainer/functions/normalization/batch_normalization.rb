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
          @decay = decay
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x, gamma, beta = inputs
          xp = Chainer.get_array_module(x)

          if @running_mean.nil?
            @running_mean = xp::NArray[*gamma].new_zeros
            @running_var = xp::NArray[*gamma].new_zeros
          end

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
          @mean = x.mean(axis: @axis)

          # TODO: Numo::Array can not be specified standard deviation
          var = ((x - x.mean(axis: @axis, keepdims: true)) ** 2).mean(axis: @axis)

          var += @eps
          @inv_std = var ** (-0.5)

          y = apply_bn_fwd(xp, x, expander.(@mean), expander.(@inv_std), gamma, beta)
          # Update running statistics
          m = x.size.div(gamma.size)
          adjust = m / [m - 1.0, 1.0].max
          @running_mean *= @decay
          @running_mean += (1 - @decay) * @mean
          @running_var *= @decay
          @running_var += (1 - @decay) * adjust * var

          [y]
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

          f = BatchNormalizationGrad.new(@eps, @expander, @axis, @mean, @inv_std)
          f.(x, gamma, gy)
        end
      end

      class BatchNormalizationGrad < Function
        include Calculation

        def initialize(eps, expander, axis, mean, inv_std)
          @eps = eps
          @expander = expander
          @axis = axis
          @mean = mean
          @inv_std = inv_std
        end

        def forward(inputs)
          retain_inputs([0, 1, 2])
          x, gamma, gy = inputs
          expander = @expander

          inv_m = gamma.class.new.fill(1.0 / x.size.div(gamma.size))
          xp = Chainer.get_array_module(x)

          gbeta = gy.sum(axis: @axis)
          x_hat = x_hat(x, expander.(@mean), expander.(@inv_std))
          ggamma = (gy * x_hat).sum(axis: @axis)
          gx = expander.(gamma * @inv_std) * (gy - (x_hat * expander.(ggamma) + expander.(gbeta)) * inv_m)

          retain_outputs([0, 1])
          [gx, ggamma, gbeta]
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
    end
  end
end
