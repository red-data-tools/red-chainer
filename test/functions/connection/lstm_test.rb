# frozen_string_literal: true

module Chainer::Functions::Connection
  class LSTMTest < Test::Unit::TestCase
    data = Chainer::Testing::Parameterize.product_dict(
      [
        {in_size: 10, out_size: 10},
        {in_size: 10, out_size: 40},
      ], [
        {input_none: false},
        {input_none: true, input_omit: true},
        {input_none: true, input_omit: false},
      ], [
        {input_variable: false},
        {input_variable: true},
      ]
    )

    def _setup(data)
      if data[:input_none]
        if data[:input_omit]
          @link = LSTM.new(data[:out_size])
        else
          @link = LSTM.new(nil, out_size: data[:out_size])
        end
      else
        @link = LSTM.new(data[:in_size], out_size: data[:out_size])
      end
      @link.cleargrads()
      x1_shape = (4, data[:in_size])
      @x1 = Numo::SFloat.new(x1_shape).rand(-1, 1)
      x2_shape = (3, data[:in_size])
      @x1 = Numo::SFloat.new(x2_shape).rand(-1, 1)
      x3_shape = (0, data[:in_size])
      @x1 = Numo::SFloat.new(x3_shape).rand(-1, 1)
    end

    data(data)
    def test_forward(data)
      x1 = data[:input_variable] ? Chainer::Variable.new(@x1) : @x1
      h1 = @link.(x1)
      c0 = Chainer::Variable.new(@x1.class.zeros(@x1.shape[0], @out_size))
      (c1_expect, h1_expect) = LSTM.lstm(c0, @link.upward(x1))

      Chainer::Testing.assert_allclose(h1.data, h1_expect.data)
      Chainer::Testing.assert_allclose(@link.h.data, h1_expect.data)
      Chainer::Testing.assert_allclose(@link.c.data, c1_expect.data)

      batch = @x2.shape[0]
      x2 = data[:input_variable] ? Chainer::Variable(@x2) : @x2
      (h1_in, h1_rest) = Chainer::Functions::Array::SplitAxis.split_axis(@link.h.data, [batch], axis: 0)
      y2 = @link.(x2)
      (c2_expect, y2_expect) = LSTM.lstm(c1_expect, @link.upward(x2) + @link.lateral(h1_in))

      Chainer::Testing.assert_allclose(y2.data, y2_expect.data)
      Chainer::Testing.assert_allclose(@link.h.data[0...batch, true], y2_expect.data)
      Chainer::Testing.assert_allclose(@link.h.data[batch...@link.h.shape[0], true], h1_rest.data)

      x3 = data[:input_variable] ? Chainer::Variable.new(@x3) : @x3
      h2_rest = @link.h
      y3 = @link(x3)
      (c3_expect, y3_expect) = LSTM.lstm(c2_expect, @link.upward(x3))
      Chainer::Testing.assert_allclose(y3.data, y3_expect.data)
      Chainer::Testing.assert_allclose(@link.h.data, h2_rest.data)
    end
  end
end
