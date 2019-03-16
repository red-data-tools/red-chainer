# frozen_string_literal: true

require 'chainer/functions/array/rollaxis'

class Chainer::Functions::Array::RollaxisTest < Test::Unit::TestCase
  data(:test_case, [
    {axis: 0, start: 2, out_shape: [3, 2, 4]},
    {axis: 2, start: 0, out_shape: [4, 2, 3]},
    {axis: 1, start: 1, out_shape: [2, 3, 4]},
    {axis: -3, start: 2, out_shape: [3, 2, 4]},
    {axis: -1, start: 0, out_shape: [4, 2, 3]},
    {axis: -2, start: -2, out_shape: [2, 3, 4]},
    {axis: 0, start: 3, out_shape: [3, 4, 2]},
    {axis: 2, start: -3, out_shape: [4, 2, 3]},
  ], keep: true)

  def setup
    @dtype = Numo::SFloat

    @axis = data[:test_case][:axis]
    @start = data[:test_case][:start]
    @out_shape = data[:test_case][:out_shape]

    @x = @dtype.new([2, 3, 4]).rand(-1, 1)
    @g = @dtype.new(@out_shape).rand(-1, 1)
    @gg = @dtype.new([2, 3, 4]).rand(-1, 1)
  end

  def check_forward(x_data)
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Array::Rollaxis.rollaxis(x, @axis, start: @start)

    expect = Chainer::Utils::Array.rollaxis(@x, @axis, start: @start)
    Chainer::Testing.assert_allclose(expect, y.data)
  end

  def test_forward
    check_forward(@x)
  end

  def check_backward(x_data, g_data)
    func = -> (x) do
      Chainer::Functions::Array::Rollaxis.rollaxis(x, @axis, start: @start)
    end

    Chainer::check_backward(func, x_data, g_data, dtype: xm::DFloat)
  end

  def test_backward
    check_backward(@x, @g)
  end

  def check_double_backward(x_data, g_data, gg_data)
    func = -> (x) do
      y = Chainer::Functions::Array::Rollaxis.rollaxis(x, @axis, start: @start)
      y * y
    end

    Chainer::check_double_backward(func, x_data, g_data, gg_data)
  end

  def test_double_backward_cpu
    check_double_backward(@x, @g, @gg)
  end
end
