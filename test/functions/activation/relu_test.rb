# frozen_string_literal: true

require 'chainer/functions/activation/relu'

class Chainer::Functions::Activation::ReLUTest < Test::Unit::TestCase
  data = {
    'test1' => {shape: [3, 2], dtype: Numo::SFloat},
    'test2' => {shape: [], dtype: Numo::SFloat},
    'test3' => {shape: [3, 2], dtype: Numo::DFloat},
    'test4' => {shape: [], dtype: Numo::DFloat}}

  def _setup(data)
    # Avoid unstability of numerical grad
    @shape = data[:shape]
    @dtype = data[:dtype]

    @dtype.srand(1) # To avoid false of "nearly_eq().all?", Use fixed seed value.
    @x = @dtype.new(@shape).rand(2) - 1
    @shape.map do |x|
      if (-0.1 < x) and (x < 0.1)
        0.5
      else
        x
      end
    end
    @gy = @dtype.new(@shape).rand(2) - 1
    @check_backward_options = {}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::Relu.relu(x)
    assert_equal(@dtype, y.data.class)
    expected = @x.dup
    if expected.shape == []
      expected[expected < 0] = 0
    else
      @x.each_with_index do |x, *i|
        if x < 0
          expected[*i] = 0
        end
      end
    end
    assert_true(y.data.nearly_eq(expected).all?)
  end

  data(data)
  def test_forward_cpu(data)
    _setup(data)
    check_forward(@x.dup)
  end
end
