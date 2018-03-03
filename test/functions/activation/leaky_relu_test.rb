# frozen_string_literal: true

require 'chainer/functions/activation/leaky_relu'

class Chainer::Functions::Activation::LeakyReLUTest < Test::Unit::TestCase
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
      if (-0.05 < x) and (x < 0.05)
        0.5
      else
        x
      end
    end
    @gy = @dtype.new(@shape).rand(2) - 1
    @slope = Random.rand
    @check_forward_options = {}
    @check_backward_options = {dtype: Numo::DFloat}
  end

  def check_forward(x_data)
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::LeakyReLU.leaky_relu(x, slope: @slope)
    assert_equal(@dtype, y.data.class)
    expected = @x.dup
    if expected.shape == []
      expected[expected < 0] *= @slope
    else
      @x.each_with_index do |x, *i|
        if x < 0
          expected[*i] *= @slope
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

  def check_backward(x_data, y_grad)
      Chainer::check_backward(Chainer::Functions::Activation::LeakyReLU.new(slope: @slope), x_data, y_grad, @check_backward_options)
  end

  data(data)
  def test_backward_cpu(data)
    _setup(data)
    check_backward(@x.dup, @gy.dup)
  end
end
