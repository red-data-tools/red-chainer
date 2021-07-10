# frozen_string_literal: true

require 'numo/narray'
require 'chainer'
require 'chainer/functions/activation/log_softmax'

class Chainer::Functions::Activation::LogSoftmaxTest < Test::Unit::TestCase

  data(:shape, [nil, [2, 3], [2, 2, 3], [2, 2, 2, 3]], keep: true)
  data(:dtype, [xm::SFloat, xm::DFloat],               keep: true)

  def setup
    @shape = data[:shape]
    @dtype = data[:dtype]
    if @shape.nil?
      value = -1000
      @x = @dtype.cast([[value, 1]])
    else
      @dtype.srand(1) # To avoid false of "assert_allclose", Use fixed seed value.
      @x = @dtype.new(@shape).rand(2) - 1
    end
    @gy = @dtype.new(@x.shape).rand(2) - 1
    @ggx = @dtype.new(@x.shape).rand(2) - 1
    @check_forward_options = {}
    @check_backward_options = {dtype: xm::DFloat}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::LogSoftmax.log_softmax(x).dup
    assert_equal(@dtype, y.data.class)

    xm = Chainer.get_array_module(@x)
    log_z = xm::NMath.log(xm::NMath.exp(@x).sum(axis:1, keepdims:true))
    y_expect = @x - log_z
    Chainer::Testing.assert_allclose(y.data, y_expect)
  end

  def test_forward
    check_forward(@x.dup)
  end

  def check_backward(x_data, gy_data, use_cudnn: "always")
    Chainer::check_backward(Chainer::Functions::Activation::LogSoftmax.method(:log_softmax), x_data, gy_data, **@check_backward_options)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, gy_data, ggx_data, use_cudnn: 'always')
    Chainer::check_double_backward(Chainer::Functions::Activation::LogSoftmax.method(:log_softmax), x_data, gy_data, ggx_data, **@check_backward_options)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
