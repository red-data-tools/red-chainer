# frozen_string_literal: true

require 'chainer/functions/array/select_item'

class Chainer::Functions::Array::SelectItemTest < Test::Unit::TestCase
  data(:test_case, [
    {in_shape: [10, 5], out_shape: [10]},
    {in_shape: [0, 5],  out_shape: [0]},
    {in_shape: [1, 33], out_shape: [1]},
    {in_shape: [10, 5], out_shape: [10]},
    {in_shape: [10, 5], out_shape: [10]},
  ], keep: true)
  data(:dtype,     [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @dtype = data[:dtype]
    @in_shape = data[:test_case][:in_shape]
    @out_shape = data[:test_case][:out_shape]

    @x_data = @dtype.new(@in_shape).rand(-1, 1)
    @t_data = xm::Int32.new(@out_shape).rand(0, 2)
    @gy_data = @dtype.new(@out_shape).rand(-1, 1)
    @ggx_data = @dtype.new(@in_shape).rand(-1, 1)

    @check_backward_options = {atol: 0.01, rtol: 0.01}
  end

  def check_forward(x_data, t_data)
    x = Chainer::Variable.new(x_data)
    t = Chainer::Variable.new(t_data)
    y = Chainer::Functions::Array::SelectItem.select_item(x, t)

    y_exp = x_data.class.zeros(t_data.size)
    t_data.size.times.each do |i|
      y_exp[i] = x_data[i, t_data[i]]
    end

    assert_equal(@dtype, y.data.class)
    assert_equal(y_exp.to_a, y.data.to_a)
  end

  def test_forward
    check_forward(@x_data, @t_data)
  end

  def check_backward(x_data, t_data, gy_data)
    func = -> (x, t) do
      Chainer::Functions::Array::SelectItem.select_item(x, t)
    end
    Chainer::check_backward(func, [x_data, t_data], gy_data, dtype: xm::DFloat, eps: 0.01, **@check_backward_options)
  end

  def test_backward
    check_backward(@x_data, @t_data, @gy_data)
  end

  def check_double_backward(x_data, t_data, gy_data, ggx_data)
    func = -> (x) do
      y = Chainer::Functions::Array::SelectItem.select_item(x, t_data)
      y * y
    end

    Chainer::check_double_backward(func, x_data, gy_data, ggx_data, eps: 0.01, dtype: xm::DFloat, **@check_backward_options)
  end

  def test_double_backward
    check_double_backward(@x_data, @t_data, @gy_data, @ggx_data)
  end
end
