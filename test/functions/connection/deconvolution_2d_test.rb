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

    @w = data[:w_dtype].new(in_channels, out_channels, kh, kw).rand
    @b = data[:nobias] ? nil : data[:x_dtype].new(out_channels).rand

    n = 2
    inh, inw = 4, 3
    outh = Chainer::Utils::Conv.get_deconv_outsize(inh, kh, sh, ph)
    outw = Chainer::Utils::Conv.get_deconv_outsize(inw, kw, sw, pw)

    @outsize = data[:test_outsize] ? [outh, outw] : nil
    @x = data[:x_dtype].new(n, in_channels, inh, inw).rand
    @gy = data[:x_dtype].new(n, out_channels, outh, outw).rand

    @ggx = data[:x_dtype].new(*@x.shape).rand
    @ggw = data[:w_dtype].new(*@w.shape).rand
    @ggb = data[:nobias] ? nil : data[:x_dtype].new(*@b.shape).rand

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
end

