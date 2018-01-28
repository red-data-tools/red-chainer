# frozen_string_literal: true

require 'chainer/functions/activation/sigmoid'

class Chainer::Functions::Activation::SigmoidTest < Test::Unit::TestCase
  data = {
    'test1' => {shape: [3, 2], dtype: Numo::SFloat},
    'test2' => {shape: [], dtype: Numo::SFloat},
    'test3' => {shape: [3, 2], dtype: Numo::DFloat},
    'test4' => {shape: [], dtype: Numo::DFloat}}

  def _setup(data)
    @shape = data[:shape]
    @dtype = data[:dtype]
    @dtype.srand(1) # To avoid false of "nearly_eq().all?", Use fixed seed value.
    @x = @dtype.new(@shape).rand(1) - 0.5
    @gy = @dtype.new(@shape).rand(0.2) - 0.1
    @check_forward_options = {}
    @check_backward_options = {}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::Sigmoid.sigmoid(x)
    assert_equal(@dtype, y.data.class)
    y_expect = Chainer::Functions::Activation::Sigmoid.sigmoid(Chainer::Variable.new(@x))
    assert_true(y.data.nearly_eq(y_expect.data).all?)
  end

  data(data)
  def test_forward_cpu(data)
    _setup(data)
    check_forward(@x.dup)
  end
end
