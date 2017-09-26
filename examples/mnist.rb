require 'chainer'
require 'fileutils'
require 'tmpdir'

class MLP < Chainer::Chain
  L = Chainer::Links::Connection::Linear
  R = Chainer::Functions::Activation::Relu

  def initialize(n_units, n_out)
    super()
    init_scope do
      @l1 = L.new(nil, out_size: n_units)
      @l2 = L.new(nil, out_size: n_units)
      @l3 = L.new(nil, out_size: n_out)
    end
  end

  def call(x)
    h1 = R.relu(@l1.(x))
    h2 = R.relu(@l2.(h1))
    @l3.(h2)
  end
end

model = Chainer::Links::Model::Classifier.new(MLP.new(1000, 10))

optimizer = Chainer::Optimizers::Adam.new
optimizer.setup(model)
train, test = Chainer::Datasets::Mnist.get_mnist

train_iter = Chainer::Iterators::SerialIterator.new(train, 100)
test_iter = Chainer::Iterators::SerialIterator.new(test, 100, repeat: false, shuffle: false)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1)
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [20, 'epoch'], out: 'result')

trainer.extend(Chainer::Training::Extensions::Evaluator.new(test_iter, model, device: -1))
trainer.extend(Chainer::Training::Extensions::LogReport.new)
trainer.extend(Chainer::Training::Extensions::PrintReport.new(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(Chainer::Training::Extensions::ProgressBar.new)

trainer.run
