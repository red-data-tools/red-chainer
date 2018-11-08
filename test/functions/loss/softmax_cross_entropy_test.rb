# frozen_string_literal: true

require 'chainer/functions/loss/softmax_cross_entropy'

class TestSoftmaxCrossEntropy < Test::Unit::TestCase
  shape        = [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]]
  cache_score  = [true, false]
  normalize    = [true, false]
  ignore_index = [nil, false, [0], [0, 1], [0, 1, 0]]
  dtype        = [Numo::SFloat]
  weight_apply = [false, true]

  value1 = shape.product(cache_score, normalize, ignore_index, dtype, weight_apply).collect {|v|
    {shape: v[0], cache_score: v[1], normalize: v[2], ignore_index: v[3], dtype: v[4], weight_apply: v[5]} }

  shape        = [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]]
  cache_score  = [false]
  normalize    = [true]
  ignore_index = [[0, 1]]
  dtype        = [Numo::SFloat, Numo::DFloat]
  weight_apply = [false, true]

  value2 = shape.product(cache_score, normalize, ignore_index, dtype, weight_apply).collect {|v|
    {shape: v[0], cache_score: v[1], normalize: v[2], ignore_index: v[3], dtype: v[4], weight_apply: v[5]} }
  value = value1 + value2
  data = (1..value.size).to_a.zip(value).to_h

  def _setup(data)
    @shape        = data[:shape]
    @cache_score  = data[:cache_score]
    @normalize    = data[:normalize]
    @ignore_index = data[:ignore_index]
    @dtype        = data[:dtype]
    @weight_apply = data[:weight_apply]

    if @shape.nil?
      @x = @dtype.cast([[-1000, 1]])
      @t = Numo::Int32[0]
    else
      @x = @dtype.new(@shape).rand(2) - 1
      out_shape = [@shape[0]] + @shape[2..-1]
      @t = Numo::Int32.new(out_shape).rand(@shape[1])

      if @ignore_index && @ignore_index.size <= @t.ndim
        @t[@ignore_index] = -1
      end
    end
    @check_forward_options = {}
    @check_backward_options_dtype = Numo::DFloat

    if @weight_apply
      @class_weight = @dtype.new([@x.shape[1]]).rand(10)
    else
      @class_weight = nil
    end
  end

  def check_forward(x_data, t_data, class_weight, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    t = Chainer::Variable.new(t_data)
    loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(x, t, normalize: @normalize, cache_score: @cache_score, class_weight: class_weight)
    assert_equal([], loss.data.shape)
    assert_equal(@dtype, loss.data.class)
    assert_equal(@cache_score, loss.creator.instance_variable_defined?(:@y))
    loss_value = loss.data.to_f
    loss_expect = 0.0
    count = 0
    x = Chainer::Utils::Array.rollaxis(@x, 1, start:@x.ndim).reshape(@t.size, @x.shape[1])
    t = @t.flatten.dup
    xm = Chainer.get_array_module(x)

    (0...x.shape[0]).map{|i|[x[i, false], t[i]]}.each do |xi, ti|
      if ti == -1
        next
      end
      log_z = xm::NMath.log(xm::NMath.exp(xi).sum())
      if class_weight.nil?
        loss_expect -= (xi - log_z)[ti]
      else
        loss_expect -= (xi - log_z)[ti] * class_weight[ti]
      end
      count += 1
    end
    if @normalize
      if count == 0
        loss_expect = 0.0
      else
        loss_expect = loss_expect / count.to_f
      end
    else
      loss_expect = loss_expect / t_data.shape[0].to_f
    end
    Chainer::Testing.assert_allclose(loss_expect, loss_value)
  end

  data(data)
  def test_forward(data)
    _setup(data)
    check_forward(@x, @t, @class_weight)
  end

  def check_backward(x_data, t_data, class_weight, use_cudnn: "always")
    func =  Chainer::Functions::Loss::SoftmaxCrossEntropy.new(cache_score: @cache_score, class_weight: class_weight)
    Chainer::check_backward(func, [x_data, t_data], nil, eps: 0.02, dtype: @check_backward_options_dtype)
  end

  data(data)
  def test_backward(data)
    _setup(data)
    check_backward(@x, @t, @class_weight)
  end
end

class TestClassWeightAssertion < Test::Unit::TestCase
  def _setup()
    @x = Numo::NArray[[0, 1], [2, 3]]
    @t = Numo::NArray[0, 1]
  end

  def test_ndim_assertion()
    _setup()
    wrong_ndim_class_weight = Numo::NArray.cast([[0, 0]])
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_ndim_class_weight)
    }
  end

  def test_dtype_assertion()
    _setup()
    wrong_dtype_class_weight = Numo::Int32.cast([0, 0])
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_dtype_class_weight)
    }
  end

  def test_variable_assertion()
    _setup()
    wrong_inst_class_weight = Chainer::Variable.new(Numo::NArray.cast([0, 0]))
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_inst_class_weight)
    }
  end
end

class TestElementwiseSoftmaxCrossEntropy < Test::Unit::TestCase
  shape        = [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]]
  cache_score  = [true, false]
  normalize    = [true, false]
  ignore_index = [nil, false, [0], [0, 1], [0, 1, 0]]
  dtype        = [Numo::SFloat]
  weight_apply = [false, true]

  value1 = shape.product(cache_score, normalize, ignore_index, dtype, weight_apply).collect {|v|
    {shape: v[0], cache_score: v[1], normalize: v[2], ignore_index: v[3], dtype: v[4], weight_apply: v[5]} }

  shape        = [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]]
  cache_score  = [false]
  normalize    = [true]
  ignore_index = [[0, 1]]
  dtype        = [Numo::SFloat, Numo::DFloat]
  weight_apply = [false, true]

  value2 = shape.product(cache_score, normalize, ignore_index, dtype, weight_apply).collect {|v|
    {shape: v[0], cache_score: v[1], normalize: v[2], ignore_index: v[3], dtype: v[4], weight_apply: v[5]} }
  value = value1 + value2
  data = (1..value.size).to_a.zip(value).to_h

  def _setup(data)
    @shape        = data[:shape]
    @cache_score  = data[:cache_score]
    @normalize    = data[:normalize]
    @ignore_index = data[:ignore_index]
    @dtype        = data[:dtype]
    @weight_apply = data[:weight_apply]

    if @shape.nil?
      @x = @dtype[[-1000, 1]]
      @t = Numo::Int32.cast([0])
    else
      @x = @dtype.new(@shape).rand(2) - 1
      out_shape = [@shape[0]] + (@shape[2..-1])
      @t = Numo::Int32.new(out_shape).rand(@shape[1])
      if @ignore_index && @ignore_index.size <= @t.ndim
        @t[@ignore_index] = -1
      end
    end

    @g = @dtype.new(@t.shape).rand(2) - 1
    @check_forward_options = {}
    @check_backward_options_dtype = Numo::DFloat

    if @weight_apply
      @class_weight = @dtype.new([@x.shape[1]]).rand(10)
    else
      @class_weight = nil
    end
  end

  def check_forward(x_data, t_data, class_weight)
    x = Chainer::Variable.new(x_data)
    t = Chainer::Variable.new(t_data)
    loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(x, t, cache_score: @cache_score, class_weight: class_weight, reduce: "no")

    assert_equal(t_data.shape, loss.shape)
    assert_equal(@dtype, loss.data.class)
    assert_equal(@cache_score, loss.creator.instance_variable_defined?(:@y))
    loss_value = loss.data

    x = Chainer::Utils::Array.rollaxis(@x, 1, start:@x.ndim).reshape(@t.size, @x.shape[1])
    t = @t.flatten.dup
    l = loss_value.flatten.dup
    xm = Chainer.get_array_module(x)

    (0...x.shape[0]).map{|i|[x[i, false], t[i], l[i]]}.each do |xi, ti, li|
      if ti == -1
        next
      end
      log_z = xm::NMath.log(xm::NMath.exp(xi).sum())

      if class_weight.nil?
        loss_expect = -(xi - log_z)[ti]
      else
        loss_expect = -(xi - log_z)[ti] * class_weight[ti]
      end
      Chainer::Testing.assert_allclose(loss_expect, li)
    end
  end

  data(data)
  def test_forward(data)
    _setup(data)
    check_forward(@x, @t, @class_weight)
  end

  def check_backward(x_data, t_data, g_data, class_weight)
    func = Chainer::Functions::Loss::SoftmaxCrossEntropy.new(cache_score: @cache_score, class_weight: class_weight, reduce: "no")
    Chainer::check_backward(func, [x_data, t_data], g_data, eps: 0.02,  dtype: @check_backward_options_dtype)
  end

  data(data)
  def test_backward(data)
    _setup(data)
    check_backward(@x, @t, @g, @class_weight)
  end
end

class TestSoftmaxCrossEntropyInvalidReduce < Test::Unit::TestCase
  use_cudnn    = ['always', 'auto', 'never'],
  normalize    = [true, false]
  cache_score  = [true, false]

  data = use_cudnn.product(normalize, cache_score).collect {|v|
    {use_cudnn: v[0], normalize: v[1], cache_score: v[2]}}
  data = (1..data.size).to_a.zip(data).to_h

  def _setup(data)
    @use_cudnn    = data[:use_cudnn]
    @normalize    = data[:normalize]
    @cache_score  = data[:cache_score]

    @x = Numo::SFloat.new([2, 3]).rand(2) - 1
    @t = Numo::Int32.zeros([2])
  end

  def check_invalid_reduce(x, t)
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(x, t, normalize: @normalize, cache_score: @cache_score, reduce: "unknown_reduce_type")
    }
  end

  data(data)
  def test_invalid_reduce(data)
    _setup(data)
    check_invalid_reduce(@x, @t)
  end
end

class TestNonDefaultIgnoreLabel < Test::Unit::TestCase
  reduce       = ['mean', 'no']
  class_weight = [nil, Numo::SFloat.ones([3])]

  data = reduce.product(class_weight).collect {|v|
    {reduce: v[0], class_weight: v[1]}}
  data = (1..data.size).to_a.zip(data).to_h

  def _setup(data)
    @reduce       = data[:reduce]
    @class_weight = data[:class_weight]

    @ignore_label = -2
    @x = Numo::SFloat.new([2, 3]).rand(2) - 1
    @t = Numo::Int32.new([2]).fill(@ignore_label)
    if @reduce == "mean"
      gy_shape = []
    else
      gy_shape = [2]
    end
    @gy = Numo::SFloat.new(gy_shape).rand(2) - 1
  end

  def check_forward(xp)
    x = xp.cast(@x)
    t = xp.cast(@t)
    if @class_weight
      class_weight = xp.asarray(@class_weight)
    else
      class_weight = nil
    end
    loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(x, t, reduce: @reduce, class_weight: class_weight, ignore_label: @ignore_label)

    if @reduce == "mean"
      expect = 0.0
    else
      expect = Numo::SFloat.zeros([2])
    end
    Chainer::Testing.assert_allclose(expect, loss.data)
  end

  data(data)
  def test_forward(data)
    _setup(data)
    check_forward(Numo::NArray)
  end

  def check_backward(xp)
    x = xp.cast(@x)
    t = xp.cast(@t)
    gy = xp.cast(@gy)
    if @class_weight
      class_weight = xp.cast(@class_weight)
    else
      class_weight = nil
    end
    f = Chainer::Functions::Loss::SoftmaxCrossEntropy.new(reduce: @reduce, class_weight: class_weight, ignore_label: @ignore_label)
    Chainer::check_backward(f, [x, t], gy)
  end

  data(data)
  def test_backward(data)
    _setup(data)
    check_backward(Numo::NArray)
  end
end
