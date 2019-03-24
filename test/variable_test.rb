# frozen_string_literal: true

require 'chainer'

class Constant < Chainer::Function
  def initialize(outputs)
    @outputs = outputs
  end
  def forward(inputs)
    return @outputs
  end

  def backward(inputs, grad_outputs)
    return inputs.map{|_| _.new_zeros}.to_a
  end
end

def constant(xs, value)
  return Constant.new(value).(*xs)
end

class Chainer::VariableTest < Test::Unit::TestCase
  data({'test1' => {x_shape: [10], c_shape: [2, 5], label: "(2, 5), #{xm}::SFloat"},
        'test2' => {x_shape: [],   c_shape: [1],    label: "(1), #{xm}::SFloat"}},   keep: true)

  def setup
    @x_shape = data[:x_shape]
    @label = data[:label]
    @c_shape = data[:c_shape]
    @x = xm::SFloat.new(@x_shape).rand(2) - 1
    @a = xm::SFloat.new(@x_shape).rand(9.9) + 0.1

    if @x_shape.size != 0
        @size = xm::NArray.cast(@x_shape).prod.to_i
    else
        @size = 1
    end
    @c = xm::DFloat.new(@size).seq(0).reshape(*@c_shape).cast_to(xm::SFloat)
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

  def test_attributes(data)
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

  def test_len(data)
    check_len(false)
  end

  def check_label(expected, gpu)
    c = Chainer::Variable.new(@c)
    assert_equal(expected, c.label)
  end

  def test_label(data)
    check_label(@label, false)
  end

  def check_backward(inputs, intermediates, outputs, retain_grad)
    outputs.each do |o|
      o.backward(retain_grad: retain_grad)
    end
    assert(inputs.map{|x| !x.grad.nil?}.all?)
    if retain_grad
      assert(intermediates.map{|x| !x.grad.nil?}.all?)
    else
      assert(intermediates.map{|x| x.grad.nil?}.all?)
    end
    assert(outputs.map{|x| !x.grad.nil?}.any?)
  end

  def create_linear_chain(length, gpu)
    x = Chainer::Variable.new(@x)
    ret = [x]
    length.times{|i|
      ret.push(constant([ret[i]], [@a]))
    }
    ret[-1].grad = ret[-1].data.new_zeros
    return ret
  end

  def test_backward(data)
    ret = create_linear_chain(2, false)
    check_backward([ret[0]], [ret[1]], [ret[2]], false)
  end

  def test_grad_type_check_pass
    a = Chainer::Variable.new(xm::SFloat.new([3]))
    a.grad = xm::SFloat.new([3])
  end

  def test_grad_type_check_type
    a = Chainer::Variable.new(xm::SFloat.new([]))
    #assert_raise(TypeError) { ## No Error
      a.grad = xm::SFloat.new
    #}
  end

  def test_grad_type_check_dtype
    a = Chainer::Variable.new(xm::SFloat.new([3]))
    assert_raise(TypeError) {
      a.grad = xm::DFloat.new([3])
    }
  end

  def test_grad_type_check_shape
    a = Chainer::Variable.new(xm::SFloat.new([3]))
    assert_raise(TypeError) {
      a.grad = xm::SFloat.new([2])
    }
  end

  def check_double_backprop
    xp = Chainer.get_array_module(@x)
    x = Chainer::Variable.new(@x)
    x.grad_var = nil

    y = x * x * x
    y.grad = y.data.new_ones
    y.backward(enable_double_backprop: true)
    gx = x.grad_var
    x.grad_var = nil  # clear grad
    gx.grad = x.data.new_ones
    gx.backward
    expect = 6 * x
    Chainer::Testing.assert_allclose(expect.data, x.grad_var.data)
  end

  def test_double_backprop
    check_double_backprop
  end
end

class IdentityFunction < Chainer::Function
  def forward(inputs)
    inputs
  end

  def backward(inputs, grad_outputs)
    grad_outputs
  end
end

class Chainer::TestVariableDoubleBackward < Test::Unit::TestCase
  def test_default_backward
    x = Chainer::Variable.new(xm::SFloat.new(1).rand)
    y = Chainer::Functions::Math::Identity.identity(x)
    y.backward
    assert_equal(nil, x.grad_var.creator)
    x.grad_var.backward
    assert_equal(nil, y.grad_var.grad_var)
  end

  def test_raise_double_backprop
    x = Chainer::Variable.new(xm::SFloat.new(1).rand)
    y = IdentityFunction.new.(x)
    y.backward(enable_double_backprop: true)
    assert_raise(RuntimeError) do
      x.grad_var.backward
    end
  end

  def test_raise_double_backprop_2
    x = Chainer::Variable.new(xm::SFloat.new(1).rand)
    z = Chainer::Functions::Math::Identity.identity(x)
    y = IdentityFunction.new.(z)
    y.backward(enable_double_backprop: true)
    assert_raise(RuntimeError) do
      x.grad_var.backward
    end
  end
end

class Chainer::AsVariableTest < Test::Unit::TestCase
  def test_to_variable_from_array
    x = Numo::DFloat.new(3).seq
    y = Chainer::Variable.as_variable(x)

    assert_equal(Chainer::Variable, y.class)
    assert_equal(x, y.data)
    assert_equal(false, y.requires_grad)
  end

  def test_to_variable_from_variable
    x = Numo::DFloat.new(3).seq
    x = Chainer::Variable.new(x)
    y = Chainer::Variable.as_variable(x)

    assert_equal(x, y)
    assert_equal(true, y.requires_grad)
  end
end
