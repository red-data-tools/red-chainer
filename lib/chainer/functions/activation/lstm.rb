class Chainer::Functions::Activation::LSTM < Chainer::Function

  # Long short-term memory unit with forget gate.
  #
  # It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
  # state. x must have four times channels compared to the number of units.

  def forward_cpu(inputs)
    (c_prev, x) = inputs
    (a, i, f, o) = extract_gates(x)
    batch_idx = [(0...x.shape[0]).to_a] + [true] * (c_prev.shape.size - 1)

    @a = a.class::Math.tanh(a)
    @i = sigmoid(i)
    @f = sigmoid(f)
    @o = sigmoid(o)

    c_next = c_prev.class.zeros(c_prev.shape)
    c_next[*batch_idx] = @a * @i + @f * c_prev[*batch_idx]
    h = @o * c_next.class::Math.tanh(c_next[*batch_idx])

    next_batch_idx = [(x.shape[0]...c_prev.shape[0]).to_a] + [true] * (c_prev.shape.size - 1)
    c_next[*next_batch_idx] = c_prev[*next_batch_idx]
    @c = c_next[*batch_idx]
    [c_next, h]
  end


  def backward_cpu(inputs, grad_outputs)
    (c_prev, x) = inputs
    (gc, gh) = grad_outputs

    batch_idx = [(0...x.shape[0]).to_a] + [true] * (c_prev.shape.size - 1)
    batch_idx_rest = [(x.shape[0]...c_prev.shape[0]).to_a] + [true] * (c_prev.shape.size - 1)

    (ga, gi, gf, go) = extract_gates(x)

    # Consider the case that either gradient is not given
    unless gc
      gc_update = 0
      gc_rest = 0
    else
      gc_update = gc[*batch_idx]
      gc_rest = gc[*batch_idx_rest]
    end

    gh = 0 unless gh

    co = @c.class::Math.tanh(@c)
    gc_prev = c_prev.class.zeros(c_prev.shape)
    # multiply f later
    gc_prev[*batch_idx] = gh * @o * grad_tanh(co) + gc_update
    gc = gc_prev[*batch_idx]
    ga[*([:*] * ga.shape.size)] = gc * @i * grad_tanh(@a)
    gi[*([:*] * ga.shape.size)] = gc * @a * grad_sigmoid(@i)
    gf[*([:*] * ga.shape.size)] = gc * c_prev[*batch_idx] * grad_sigmoid(@f)
    go[*([:*] * ga.shape.size)] = gh * co * grad_sigmoid(@o)
    gc_prev[*batch_idx] *= @f  # multiply f here
    gc_prev[*batch_idx_rest] = gc_rest

    r = x.reshape(x.shape[0], x.shape[1] / 4, 4, x.shape[2])
    r[true, true, 0, true] = ga
    r[true, true, 1, true] = gi
    r[true, true, 2, true] = gf
    r[true, true, 3, true] = go
    gx = r.reshape(r.shape[0], r.shape[1] * 4, r.shape[3])

    [gc_prev, gx]
  end

  def self.lstm(c_priv, x)
    self.new().(c_priv, x)
  end

private
  def extract_gates(x)
    if x.shape[0] == 0
      return [x.class.new(0, x.shape[1] / 4, x.shape[2])] * 4
    end

    r = x.reshape(x.shape[0], x.shape[1] / 4, 4, x.shape[2])
    4.times.map{ |i| r[true, true, i, true] }
  end

  def sigmoid(x)
    half = 0.5
    x.class::Math.tanh(x * half) * half + half
  end

  def grad_sigmoid(x)
    x * (1 - x)
  end

  def grad_tanh(x)
    1 - x * x
  end
end
