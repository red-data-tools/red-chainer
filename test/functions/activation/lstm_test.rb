# frozen_string_literal: true

class Chainer::Functions::Activation::LstmTest < Test::Unit::TestCase
  batches = [[2], [3]]
  dtypes = [[Numo::SFloat], [Numo::SFloat, Numo::DFloat]]

  data = batches.zip(dtypes).reduce([]) do |acc, (batch, dtype)|
    acc + batch.product(dtype).map do |b, t|
      ["#{t}: batch: #{b}", {batch: b, dtype: t}]
    end
  end.to_h

  def sigmoid(x)
    half = 0.5
    x.class::Math.tanh(x * half) * half + half
  end

  def _setup(data)
    hidden_shape = [3, 2, 4]
    x_shape = [data[:batch], 8, 4]
    y_shape = [data[:batch], 2, 4]

    @c_prev = data[:dtype].new(hidden_shape).rand(-1, 1)
    @x = data[:dtype].new(x_shape).rand(-1, 1)

    @gc = data[:dtype].new(hidden_shape).rand(-1, 1)
    @gh = data[:dtype].new(y_shape).rand(-1, 1)
  end

  data(data)
  def test_forward_cpu(data)
    _setup(data)

    c_prev = Chainer::Variable.new(@c_prev)
    x = Chainer::Variable.new(@x)
    c, h = Chainer::Functions::Activation::LSTM.lstm(c_prev, x)
    assert_equal(c.data.class, data[:dtype])
    assert_equal(h.data.class, data[:dtype])
    batch = x.shape[0]

    # Compute expected out
    a_in = @x[true, [0, 4], true]
    i_in = @x[true, [1, 5], true]
    f_in = @x[true, [2, 6], true]
    o_in = @x[true, [3, 7], true]

    batch_indices = [(0...batch).to_a] + [true] * (@c_prev.shape.size - 1)
    c_expect = sigmoid(i_in) * a_in.class::Math.tanh(a_in) + sigmoid(f_in) * @c_prev[*batch_indices]
    h_expect = sigmoid(o_in) * c_expect.class::Math.tanh(c_expect)

    Chainer::Testing.assert_allclose(c_expect, c.data[*batch_indices])
    Chainer::Testing.assert_allclose(h_expect, h.data)

    next_batch_indices = [(batch...@c_prev.shape.first).to_a] + [true] * (@c_prev.shape.size - 1)
    Chainer::Testing.assert_allclose(@c_prev[*next_batch_indices], c.data[*next_batch_indices])
  end

  data(data)
  def test_full_backward_cpu(data)
    _setup(data)
    check_backward(@c_prev, @x, @gc, @gh)
  end

  def check_backward(c_prev_data, x_data, c_grad, h_grad)
    Chainer.check_backward(
      Chainer::Functions::Activation::LSTM.new(),
      [c_prev_data, x_data],
      [c_grad, h_grad]
    )
  end
end
