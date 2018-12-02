# frozen_string_literal: true

require 'chainer/functions/loss/softmax_cross_entropy'

class TestSoftmaxCrossEntropy < Test::Unit::TestCase
  data(:shape,        [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]], group: :sfloat, keep: true)
  data(:cache_score,  [true, false],                          group: :sfloat, keep: true)
  data(:normalize,    [true, false],                          group: :sfloat, keep: true)
  data(:ignore_index, [nil, false, [0], [0, 1], [0, 1, 0]],   group: :sfloat, keep: true)
  data(:dtype,        [xm::SFloat],                           group: :sfloat, keep: true)
  data(:weight_apply, [false, true],                          group: :sfloat, keep: true)

  data(:shape,        [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]], group: :general, keep: true)
  data(:cache_score,  [false],                                group: :general, keep: true)
  data(:normalize,    [true],                                 group: :general, keep: true)
  data(:ignore_index, [[0, 1]],                               group: :general, keep: true)
  data(:dtype,        [xm::SFloat, xm::DFloat],               group: :general, keep: true)
  data(:weight_apply, [false, true],                          group: :general, keep: true)

  def setup
    @shape        = data[:shape]
    @cache_score  = data[:cache_score]
    @normalize    = data[:normalize]
    @ignore_index = data[:ignore_index]
    @dtype        = data[:dtype]
    @weight_apply = data[:weight_apply]

    if @shape.nil?
      @x = @dtype.cast([[-1000, 1]])
      @t = xm::Int32[0]
    else
      @x = @dtype.new(@shape).rand(2) - 1
      out_shape = [@shape[0]] + @shape[2..-1]
      @t = xm::Int32.new(out_shape).rand(@shape[1])

      if @ignore_index && @ignore_index.size <= @t.ndim
        @t[@ignore_index] = -1
      end
    end
    @check_forward_options = {}
    @check_backward_options_dtype = xm::DFloat

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

    (0...x.shape[0]).map{|i|[x[i, false], t[i].to_i]}.each do |xi, ti|
      if ti == -1
        next
      end
      log_z = xm::NMath.log(xm::NMath.exp(xi).sum)
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

  def test_forward
    check_forward(@x, @t, @class_weight)
  end

  def check_backward(x_data, t_data, class_weight, use_cudnn: "always")
    func =  Chainer::Functions::Loss::SoftmaxCrossEntropy.new(cache_score: @cache_score, class_weight: class_weight)
    Chainer::check_backward(func, [x_data, t_data], nil, eps: 0.02, dtype: @check_backward_options_dtype)
  end

  def test_backward
    check_backward(@x, @t, @class_weight)
  end
end

class TestClassWeightAssertion < Test::Unit::TestCase
  def setup
    @x = xm::NArray[[0, 1], [2, 3]]
    @t = xm::NArray[0, 1]
  end

  def test_ndim_assertion
    wrong_ndim_class_weight = xm::NArray.cast([[0, 0]])
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_ndim_class_weight)
    }
  end

  def test_dtype_assertion
    wrong_dtype_class_weight = xm::Int32.cast([0, 0])
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_dtype_class_weight)
    }
  end

  def test_variable_assertion
    wrong_inst_class_weight = Chainer::Variable.new(xm::NArray.cast([0, 0]))
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, class_weight: wrong_inst_class_weight)
    }
  end
end

class TestElementwiseSoftmaxCrossEntropy < Test::Unit::TestCase
  data(:shape,        [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]], group: :sfloat, keep: true)
  data(:cache_score,  [true, false],                          group: :sfloat, keep: true)
  data(:normalize,    [true, false],                          group: :sfloat, keep: true)
  data(:ignore_index, [nil, false, [0], [0, 1], [0, 1, 0]],   group: :sfloat, keep: true)
  data(:dtype,        [xm::SFloat],                           group: :sfloat, keep: true)
  data(:weight_apply, [false, true],                          group: :sfloat, keep: true)

  data(:shape,        [nil, [2, 3], [2, 3, 2], [2, 3, 2, 2]], group: :general, keep: true)
  data(:cache_score,  [false],                                group: :general, keep: true)
  data(:normalize,    [true],                                 group: :general, keep: true)
  data(:ignore_index, [[0, 1]],                               group: :general, keep: true)
  data(:dtype,        [xm::SFloat, xm::DFloat],               group: :general, keep: true)
  data(:weight_apply, [false, true],                          group: :general, keep: true)

  def setup
    @shape        = data[:shape]
    @cache_score  = data[:cache_score]
    @normalize    = data[:normalize]
    @ignore_index = data[:ignore_index]
    @dtype        = data[:dtype]
    @weight_apply = data[:weight_apply]

    if @shape.nil?
      @x = @dtype[[-1000, 1]]
      @t = xm::Int32.cast([0])
    else
      @x = @dtype.new(@shape).rand(2) - 1
      out_shape = [@shape[0]] + (@shape[2..-1])
      @t = xm::Int32.new(out_shape).rand(@shape[1])
      if @ignore_index && @ignore_index.size <= @t.ndim
        @t[@ignore_index] = -1
      end
    end

    @g = @dtype.new(@t.shape).rand(2) - 1
    @check_forward_options = {}
    @check_backward_options_dtype = xm::DFloat

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

    (0...x.shape[0]).map{|i|[x[i, false], t[i].to_i, l[i]]}.each do |xi, ti, li|
      if ti == -1
        next
      end
      log_z = xm::NMath.log(xm::NMath.exp(xi).sum)

      if class_weight.nil?
        loss_expect = -(xi - log_z)[ti]
      else
        loss_expect = -(xi - log_z)[ti] * class_weight[ti]
      end
      Chainer::Testing.assert_allclose(loss_expect, li)
    end
  end

  def test_forward
    check_forward(@x, @t, @class_weight)
  end

  def check_backward(x_data, t_data, g_data, class_weight)
    func = Chainer::Functions::Loss::SoftmaxCrossEntropy.new(cache_score: @cache_score, class_weight: class_weight, reduce: "no")
    Chainer::check_backward(func, [x_data, t_data], g_data, eps: 0.02,  dtype: @check_backward_options_dtype)
  end

  def test_backward
    check_backward(@x, @t, @g, @class_weight)
  end
end

class TestSoftmaxCrossEntropyInvalidReduce < Test::Unit::TestCase
  data(:use_cudnn,   ['always', 'auto', 'never'], group: :reduce, keep: true)
  data(:normalize,   [true, false],               group: :reduce, keep: true)
  data(:cache_score, [true, false],               group: :reduce, keep: true)

  def setup
    @use_cudnn    = data[:use_cudnn]
    @normalize    = data[:normalize]
    @cache_score  = data[:cache_score]

    @x = xm::SFloat.new([2, 3]).rand(2) - 1
    @t = xm::Int32.zeros([2])
  end

  def check_invalid_reduce(x, t)
    assert_raise(ArgumentError) {
      Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(x, t, normalize: @normalize, cache_score: @cache_score, reduce: "unknown_reduce_type")
    }
  end

  def test_invalid_reduce
    check_invalid_reduce(@x, @t)
  end
end

class TestNonDefaultIgnoreLabel < Test::Unit::TestCase
  data(:reduce,       ['mean', 'no'],              group: :reduce, keep: true)
  data(:class_weight, [nil, xm::SFloat.ones([3])], group: :reduce, keep: true)

  def setup
    @reduce       = data[:reduce]
    @class_weight = data[:class_weight]

    @ignore_label = -2
    @x = xm::SFloat.new([2, 3]).rand(2) - 1
    @t = xm::Int32.new([2]).fill(@ignore_label)
    if @reduce == "mean"
      gy_shape = []
    else
      gy_shape = [2]
    end
    @gy = xm::SFloat.new(gy_shape).rand(2) - 1
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
      expect = xm::SFloat.zeros([2])
    end
    Chainer::Testing.assert_allclose(expect, loss.data)
  end

  def test_forward
    check_forward(xm::NArray)
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

  def test_backward
    check_backward(xm::NArray)
  end
end

class TestNilIgnoreLabel < Test::Unit::TestCase
  def setup
    @reduce       = 'no'
    @class_weight = nil
    @ignore_label = nil
    @x = xm::SFloat.ones([2, 3]) * 10
    @t = xm::Int32.new([2]).fill(-1)
    @gy = xm::SFloat.ones([2])
  end

  def check_forward
    loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@x, @t, reduce: @reduce, class_weight: @class_weight, ignore_label: @ignore_label)

    expect = xm::SFloat[1.09861, 1.09861]
    Chainer::Testing.assert_allclose(expect, loss.data)
  end

  def test_forward
    check_forward
  end

  def check_backward
    f = Chainer::Functions::Loss::SoftmaxCrossEntropy.new(reduce: @reduce, class_weight: @class_weight, ignore_label: @ignore_label)
    Chainer::check_backward(f, [@x, @t], @gy, rtol: 1e-3)
  end

  def test_backward
    check_backward
  end
end
