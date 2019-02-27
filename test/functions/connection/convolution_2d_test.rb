# frozen_string_literal: true

class Chainer::Functions::Connection::Convolution2DTest < Test::Unit::TestCase
  data(:cover_all, [true, false], keep: true)
  data(:x_dtype, [Numo::SFloat], keep: true)
  data(:w_dtype, [Numo::SFloat], keep: true)

  def setup
    in_channels = 3
    out_channels = 2
    kh, kw = [3, 3]

    @stride = 2
    @pad = 1

    @w = data[:w_dtype].new(out_channels, in_channels, kh, kw).rand_norm(0, Numo::NMath.sqrt(1.0 / (kh * kw * in_channels)))
    @b = data[:x_dtype].new(out_channels).rand(-1, 1)

    @x = data[:x_dtype].new([2, 3, 4, 3]).rand(-1, 1)
    if data[:cover_all]
      @gy = data[:x_dtype].new([2, 2, 3, 2]).rand(-1 ,1)
    else
      @gy = data[:x_dtype].new([2, 2, 2, 2]).rand(-1, 1)
    end

    @ggx = data[:x_dtype].new(*@x.shape).rand(-1, 1)
    @ggw = data[:w_dtype].new(*@w.shape).rand(-1, 1)
    @ggb = data[:x_dtype].new(*@b.shape).rand(-1, 1)

    @check_forward_options = {}
    @check_backward_options = { dtype: Numo::DFloat }
    @check_double_backward_options = { dtype: Numo::DFloat }
  end

  def test_forward(data)
    x = Chainer::Variable.new(@x)
    w = Chainer::Variable.new(@w)
    b = Chainer::Variable.new(@b)

    y = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(x, w, b: b, stride: @stride, pad: @pad, cover_all: data[:cover_all])
    assert_equal(data[:x_dtype], y.data.class)
  end

  def test_backward(data)
    args = [@x, @w]
    if @b
      args << @b
    end

    func = -> (*args) do
      x, w, b = args
      Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(x, w, b: b, stride: @stride, pad: @pad, cover_all: data[:cover_all])
    end

    Chainer::check_backward(func, args, @gy, **@check_backward_options)
  end

  def test_double_backward
    args = [@x, @w]
    grad_grads = [@ggx, @ggw]
    if @b
      args << @b
      grad_grads << @ggb
    end

    func = -> (*args) do
      x, w, b = args
      y = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(x, w, b: b, stride: @stride, pad: @pad, cover_all: data[:cover_all])

      y * y  # make the function nonlinear
    end

    Chainer::check_double_backward(func, args, [@gy], grad_grads, **@check_double_backward_options)
  end
end
