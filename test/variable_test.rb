# frozen_string_literal: true

require 'chainer'

class Chainer::VariableTest < Test::Unit::TestCase
  data = {
    'test1' => {x_shape: [10], c_shape: [2, 5], label: '(2, 5), Numo::SFloat'},
    'test2' => {x_shape: [], c_shape: [1], label: '(1), Numo::SFloat'}}

  def _setup(data)
    @x_shape = data[:x_shape]
    @label = data[:label]
    @c_shape = data[:c_shape]
    @x = Numo::SFloat.new(@x_shape).rand(2) - 1

    if @x_shape.length != 0
        @size = (Numo::NArray.cast(@x_shape).prod()).to_i
    else
        @size = 1
    end
    @c = Numo::DFloat.new(@size).seq(0).reshape(*@c_shape).cast_to(Numo::SFloat)
  end

  def check_attributes(gpu)
    x = Chainer::Variable.new(@x)
    assert_equal(@x.shape, x.shape)
    assert_equal(@x.ndim, x.ndim)
    assert_equal(@x.size, x.size)
    assert_equal(@x.class, x.dtype)
    assert(x.requires_grad)
    assert(x.node.requires_grad)
  end

  data(data)
  def test_attributes_cpu(data)
    _setup(data)
    check_attributes(false)
  end

  def check_len(gpu)
    x = Chainer::Variable.new(@x)
    if x.ndim == 0
      assert_equal(1, x.size)
    else
      assert_equal(@x_shape[0], x.size)
    end
  end

  data(data)
  def test_len_cpu(data)
    _setup(data)
    check_len(false)
  end

  def check_label(expected, gpu)
    c = Chainer::Variable.new(@c)
    assert_equal(expected, c.label)
  end

  data(data)
  def test_label_cpu(data)
    _setup(data)
    check_label(@label, false)
  end
end
