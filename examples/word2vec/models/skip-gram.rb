class SkipGram < Chainer::Chain
  attr_reader :embed

  def initialize(n_vocab, n_units, loss_func)
    super()
    init_scope do
      initial_weight = Chainer::Initializers::Uniform.new(scale: (1.0 / n_units))
      @embed = Chainer::Links::Connection::EmbedID.new(n_vocab, n_units, initial_w: initial_weight)
      @loss_func = loss_func
    end
  end

  def call(args, **options)
    (x, contexts) = args
    e = @embed.(contexts)
    shape = e.shape
    x = Chainer::Functions::Array::BroadcastTo.broadcast_to(x.reshape(x.size, 1), shape[0, 2])
    e = Chainer::Functions::Array::Reshape.reshape(e, [shape[0] * shape[1], shape[2]])
    x = Chainer::Functions::Array::Reshape.reshape(x, [shape[0] * shape[1]])
    loss = @loss_func.(e, x)
    Chainer::Reporter.save_report({loss: loss}, self)
    loss
  end
end
