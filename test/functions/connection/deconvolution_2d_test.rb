# frozen_string_literal: true
require 'chainer/functions/connection/deconvolution_2d'

class Chainer::Functions::Connection::Deconvolution2DFunctionTest < Test::Unit::TestCase
  include Chainer::Functions::Connection

  data(:c_contiguous, [true], keep: true)
  data(:test_outsize, [true, false], keep: true)
  data(:nobias, [true, false], keep: true)
  data(:stride, [1, 2], keep: true)
  data(:x_dtype, [xm::SFloat, xm::DFloat], keep: true)
  data(:w_dtype, [xm::SFloat, xm::DFloat], keep: true)

  def setup
    in_channels = 3
    out_channels = 2
    ksize = 3
    @pad = 1

    kh, kw = [3, 3]
    sh, sw = data[:stride].is_a?(::Array) ? data[:stride] : [data[:stride], data[:stride]]
    ph, pw = [1, 1]

    @w = data[:w_dtype].new(in_channels, out_channels, kh, kw).rand_norm(0, Numo::NMath.sqrt(1.0 / (kh * kw * in_channels)).to_f)
    @b = data[:nobias] ? nil : data[:x_dtype].new(out_channels).rand(-1, 1)

    n = 2
    inh, inw = 4, 3
    outh = Chainer::Utils::Conv.get_deconv_outsize(inh, kh, sh, ph)
    outw = Chainer::Utils::Conv.get_deconv_outsize(inw, kw, sw, pw)

    @outsize = data[:test_outsize] ? [outh, outw] : nil
    @x = data[:x_dtype].new(n, in_channels, inh, inw).rand(-1, 1)
    @gy = data[:x_dtype].new(n, out_channels, outh, outw).rand(-1, 1)

    @ggx = data[:x_dtype].new(*@x.shape).rand(-1, 1)
    @ggw = data[:w_dtype].new(*@w.shape).rand(-1, 1)
    @ggb = data[:nobias] ? nil : data[:x_dtype].new(*@b.shape).rand(-1, 1)

    @check_backward_options = { dtype: xm::DFloat }
    @check_double_backward_options = { dtype: xm::DFloat }
  end

  def test_forward(data)
    x = Chainer::Variable.new(@x)
    w = Chainer::Variable.new(@w)
    b = data[:nobias] ? nil : Chainer::Variable.new(@b)

    y = Deconvolution2DFunction.deconvolution_2d(x, w, b: b, stride: data[:stride], pad: @pad, outsize: @outsize)

    assert_equal(data[:x_dtype], y.data.class)
  end

  def test_backward(data)
    if @b.nil?
      args = [@x, @w]
    else
      args = [@x, @w, @b]
    end

    func = -> (*args) do
      if data[:nobias]
        x, w = args
        Deconvolution2DFunction.deconvolution_2d(x, w, stride: data[:stride], pad: @pad, outsize: @outsize)
      else
        x, w, b = args
        Deconvolution2DFunction.deconvolution_2d(x, w, b: b, stride: data[:stride], pad: @pad, outsize: @outsize)
      end
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
      if data[:nobias]
        x, w = args
        y = Deconvolution2DFunction.deconvolution_2d(x, w, stride: data[:stride], pad: @pad, outsize: @outsize)
      else
        x, w, b = args
        y = Deconvolution2DFunction.deconvolution_2d(x, w, b: b, stride: data[:stride], pad: @pad, outsize: @outsize)
      end

      y * y  # make the function nonlinear
    end

    Chainer::check_double_backward(func, args, [@gy], grad_grads, **@check_double_backward_options)
  end
end
