class CBoW < Chainer::Chain
  def initialize(n_vocab, n_units, loss_func)
    super()
    init_scope do
      initial_weight = Chainer::Initializers::Uniform.new(scale: (1.0 / n_units))
      @embed = Chainer::Links::Connection::EmbedID.new(n_vocab, n_units, initial_w: initial_weight)
      @loss_func = loss_func

    end
  end

  def call(x, contexts)
    e = @embed.(contexts)
    h = F.sum(e, axis=1) * (1. / contexts.shape[1])
    loss = @loss_func.(h, x)
    loss
  end
end
