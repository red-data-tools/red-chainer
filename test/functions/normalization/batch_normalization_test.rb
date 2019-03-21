class Chainer::Functions::Normalization::BatchNormalizationFunctionTest < Test::Unit::TestCase
  def setup
    @param_shape = [3]
    @ndim = 1
    @dtype = Numo::DFloat

    @expander = -> (arr) do
      new_shape = [1] + arr.shape + [1] * @ndim
      arr.reshape(*new_shape)
    end
    @eps = 2e-5
    @decay = 0.9
    @gamma = @dtype.new(@param_shape).rand(0.5, 1)
    @beta = @dtype.new(@param_shape).rand(-1, 1)

    head_ndim = @gamma.ndim + 1
    shape = [5] + @param_shape + [2] * @ndim
    @x = @dtype.new(shape).rand(-1, 1)
    @gy = @dtype.new(shape).rand(-1, 1)
    @ggx = @dtype.new(shape).rand(-1, 1)
    @gggamma = @dtype.new(@param_shape).rand(-1, 1)
    @ggbeta = @dtype.new(@param_shape).rand(-1, 1)


    @args = [@x, @gamma, @beta]
    @ggargs = [@ggx, @gggamma, @ggbeta]
    @aggr_axes = [0] + (head_ndim...(@x.ndim)).to_a
    @mean = @x.mean(axis: @aggr_axes)
    @var = @x.var(axis: @aggr_axes) + @eps
    @train = true
    @check_forward_options = { atol: 1e-4, rtol: 1e-3}
    @check_backward_options = { dtype: xm::DFloat}
    @check_double_backward_options = { dtype: xm::DFloat, atol: 1e-3, rtol: 1e-2}
  end

  def batch_normalization(expander, gamma, beta, x, mean, var)
    mean = expander.(mean)
    std = expander.(xm::NMath.sqrt(var))
    (expander.(gamma) * (x - mean) / std + expander.(beta))
  end

  def check_forward(args, use_cudnn: 'always')
    args = args.map { |e| Chainer::Variable.new(e) }
    y = Chainer::Functions::Normalization::BatchNormalization.batch_normalization(*args, running_mean: nil, running_var: nil, decay: @decay, eps: @eps)
    assert_equal(@dtype, y.data.class)

    y_expect = batch_normalization(@expander, @gamma, @beta, @x, @mean, @var)

    Chainer::Testing.assert_allclose(y_expect, y.data, **@check_forward_options)
  end

  def test_forward
    check_forward(@args)
  end

  def check_backward(args, y_grad, use_cudnn: 'always')
    func = -> (*args) do
      Chainer::Functions::Normalization::BatchNormalization.batch_normalization(*args, decay: @decay, eps: @eps)
    end

    old_train = Chainer.configuration.train
    Chainer.configuration.train = @train
    Chainer::check_backward(func, args, y_grad, **@check_backward_options)
    Chainer.configuration.train = old_train
  end

  def test_backward
    check_backward(@args, @gy)
  end
end
