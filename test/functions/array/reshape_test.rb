# frozen_string_literal: true

require 'chainer/functions/array/reshape'

class Chainer::Functions::Array::ReshapeTest < Test::Unit::TestCase
  xm = Chainer::Device.default.xm

  data(:in_shape,  [[4, 3, 2]],              keep: true)
  data(:out_shape, [[2, 2, 6]],              keep: true)
  data(:dtype,     [xm::SFloat, xm::DFloat], keep: true)

  def test_forward()
    shape = data[:out_shape]
    in_data = data[:dtype].new(data[:in_shape]).rand(-1, 1)
    x = Chainer::Variable.new(in_data)
    y = Chainer::Functions::Array::Reshape.reshape(x, shape)
    assert_equal(y.data.class, data[:dtype])
    assert_equal(x.reshape(*shape), y.data)
  end

  def test_backward()
    in_data = data[:dtype].new(data[:in_shape]).rand(-1, 1)
    x = Chainer::Variable.new(in_data)
    y = Chainer::Functions::Array::Reshape.reshape(x, data[:out_shape])
    y.grad = y.data
    y.backward()

    Chainer::Testing.assert_allclose(x.data, x.grad, atol: 0, rtol: 0)
  end
end
