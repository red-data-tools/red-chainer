module Chainer::Functions::Activation
  # Long short-term memory unit with forget gate.
  #
  # It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
  # state. x must have four times channels compared to the number of units.
  class LSTM < Chainer::Function

    def forward_cpu(inputs)
      (c_prev, x) = inputs
      if x.shape[0] == 0
        return [c_prev, x]
      end
      (a, i, f, o) = LSTM.extract_gates(x)
      batch_idx = [(0...x.shape[0]).to_a] + [true] * (c_prev.shape.size - 1)

      @a = a.class::Math.tanh(a)
      #@a = a if a.shape[0] == 0
      @i = LSTM.sigmoid(i)
      @f = LSTM.sigmoid(f)
      @o = LSTM.sigmoid(o)

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

      gx = x.dup
      (ga, gi, gf, go) = LSTM.extract_gates(gx)

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
      gc_prev[*batch_idx] = gh * @o * LSTM.grad_tanh(co) + gc_update
      gc = gc_prev[*batch_idx]
      ga[*([:*] * gc.shape.size)] = gc * @i * LSTM.grad_tanh(@a)
      gi[*([:*] * gc.shape.size)] = gc * @a * LSTM.grad_sigmoid(@i)
      gf[*([:*] * gc.shape.size)] = gc * c_prev[*batch_idx] * LSTM.grad_sigmoid(@f)
      go[*([:*] * gc.shape.size)] = gh * co * LSTM.grad_sigmoid(@o)
      gc_prev[*batch_idx] *= @f  # multiply f here
      gc_prev[*batch_idx_rest] = gc_rest

      [gc_prev, gx]
    end

    def self.lstm(c_priv, x)
      self.new().(c_priv, x)
    end

    def self.extract_gates(x)
      # if x.shape[0] == 0
      #   return [x.class.new(0, x.shape[1]/4, *x.shape.drop(2))] * 4
      # end
      r = Numo::Int32[*(0...x.shape[1]).step(4)]
      4.times.map do |i|
        x[true, r+i, *([true] * (x.shape.size - 2))]
      end
    end

    def self.sigmoid(x)
      #return x if x.shape[0] == 0

      half = 0.5
      x.class::Math.tanh(x * half) * half + half
    end

    def self.grad_sigmoid(x)
      x * (1 - x)
    end

    def self.grad_tanh(x)
      1 - x * x
    end
  end
end
