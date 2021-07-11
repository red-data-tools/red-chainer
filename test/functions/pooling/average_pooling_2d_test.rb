class Chainer::Functions::Pooling::AveragePooling2DTest < Test::Unit::TestCase
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    @dtype = data[:dtype]

    @x = @dtype.new([2, 3, 4, 3]).rand(-1, 1)
    @gy = @dtype.new([2, 3, 2, 2]).rand(-1, 1)
    @check_forward_options = {}
    @check_backward_options = { dtype: xm::DFloat }
    @ggx = @dtype.new([2, 3, 4, 3]).rand(-1, 1)
  end

  def check_forward(x_data, use_cudnn: 'always')
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Pooling::AveragePooling2D.average_pooling_2d(x, 3, stride: 2, pad: 1)
    assert_equal(y.data.class, @dtype)

    y_data = y.data
    assert_equal(@gy.shape, y_data.shape)
    2.times.each do |k|
      3.times.each do |c|
        x = @x[k, c, false]
        expect = xm::DFloat[[x[0...2, 0...2, false].sum.to_f, x[0...2, 1...3, false].sum.to_f], [x[1...4, 0...2, false].sum.to_f, x[1...4, 1...3, false].sum.to_f]] / 9
        Chainer::Testing.assert_allclose(expect, y_data[k, c, false], **@check_forward_options)
      end
    end
  end

  def test_forward
    check_forward(@x)
  end

  def check_backward(x_data, y_grad, use_cudnn: 'always')
    func = -> (x) do
      Chainer::Functions::Pooling::AveragePooling2D.average_pooling_2d(x, 3, stride: 2, pad: 1)
    end

    Chainer::check_backward(func, x_data, y_grad, **@check_backward_options)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad, use_cudnn: 'always')
    func = -> (x) do
      y = Chainer::Functions::Pooling::AveragePooling2D.average_pooling_2d(x, 3, stride: 2, pad: 1)
      y * y
    end
    Chainer::check_double_backward(func, x_data, y_grad, x_grad_grad, **@check_backward_options)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
