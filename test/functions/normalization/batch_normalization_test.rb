class Chainer::Functions::Normalization::BatchNormalizationFunctionTest < Test::Unit::TestCase
  data(:param_shape, [[3], [3, 4], [3, 2, 3]], keep: true)
  data(:ndim, [0, 1, 2], keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @param_shape = data[:param_shape]
    @ndim = data[:ndim]
    @dtype = data[:dtype]

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
    @var = ((@x - @x.mean(axis: @aggr_axes, keepdims: true)) ** 2).mean(axis: @aggr_axes)
    @var += @eps
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

  def check_double_backward(args, y_grad, x_grad_grad, use_cudnn: 'always')
    func = -> (*args) do
      y = Chainer::Functions::Normalization::BatchNormalization.batch_normalization(*args, decay: @decay, eps: @eps)
      y * y
    end

    Chainer::check_double_backward(func, args, y_grad, x_grad_grad, **@check_double_backward_options)
  end

  def test_double_backward
    check_double_backward(@args, @gy, @ggargs)
  end
end

class Chainer::Functions::Normalization::FixedBatchNormalizationTest < Test::Unit::TestCase
  data(:param_shape, [[3], [3, 4], [3, 2, 3]], keep: true)
  data(:ndim, [0, 1, 2], keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    xm::NArray.srand

    @param_shape = data[:param_shape]
    @ndim = data[:ndim]
    @dtype = data[:dtype]

    @gamma = @dtype.new(@param_shape).rand(0.5, 1)
    @beta = @dtype.new(@param_shape).rand(-1, 1)
    @expander = -> (arr) do
      new_shape = [1] + arr.shape + [1] * @ndim
      arr.reshape(*new_shape)
    end

    shape = [5] + @param_shape + [2] * @ndim
    @x = @dtype.new(shape).rand(-1, 1)
    @eps = 2e-5
    @decay = 0.0

    head_ndim = @gamma.ndim + 1
    @aggr_axes = [0] + (head_ndim...(@x.ndim)).to_a
    @mean = @dtype.new(@param_shape).rand(-1, 1)
    @var = @dtype.new(@param_shape).rand(0.5, 1)

    @args = [@x, @gamma, @beta, @mean, @var]

    @gy = @dtype.new(shape).rand(-1, 1)
    @ggx = @dtype.new(shape).rand(-1, 1)
    @gggamma = @dtype.new(@param_shape).rand(-1, 1)
    @ggbeta = @dtype.new(@param_shape).rand(-1, 1)
    @ggmean = @dtype.new(@param_shape).rand(-1, 1)
    @ggvar = @dtype.new(@param_shape).rand(-1, 1)

    @ggargs = [@ggx, @gggamma, @ggbeta, @ggmean, @ggvar]

    @train = false
    @check_forward_options = { atol: 1e-4, rtol: 1e-3}
    @check_backward_options = { dtype: xm::DFloat }
  end

  def batch_normalization(expander, gamma, beta, x, mean, var)
    mean = expander.(mean)
    std = expander.(xm::NMath.sqrt(var))
    (expander.(gamma) * (x - mean) / std + expander.(beta))
  end

  def check_forward(args, use_cudnn: 'always')
    xes = args.map { |x| Chainer::Variable.new(x) }
    y = Chainer::Functions::Normalization::FixedBatchNormalization.fixed_batch_normalization(*xes, eps: @eps)

    assert_equal(@dtype, y.data.class)

    y_expect = batch_normalization(@expander, @gamma, @beta, @x, @mean, @var)

    Chainer::Testing.assert_allclose(y_expect, y.data, **@check_forward_options)
  end

  def test_forward
    check_forward(@args)
  end

  def check_backward(args, y_grad, use_cudnn: 'always')
    func = -> (*args) do
      Chainer::Functions::Normalization::FixedBatchNormalization.fixed_batch_normalization(*args, eps: @eps)
    end

    old_train = Chainer.configuration.train
    Chainer.configuration.train = @train
    Chainer::check_backward(func, args, y_grad, **@check_backward_options)
    Chainer.configuration.train = old_train
  end

  def test_backward
    check_backward(@args, @gy)
  end

  def check_double_backward(args, y_grad, x_grad_grad, use_cudnn: 'always')
    func = -> (*args) do
      y = Chainer::Functions::Normalization::FixedBatchNormalization.fixed_batch_normalization(*args, eps: @eps)
      y * y
    end

    Chainer::check_double_backward(func, args, y_grad, x_grad_grad, **@check_backward_options)
  end

  def test_double_backward
    check_double_backward(@args, @gy, @ggargs)
  end
end
