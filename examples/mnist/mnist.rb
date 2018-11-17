require 'chainer'
require 'fileutils'
require 'optparse'
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

args = {
  batchsize: 100,
  frequency: -1,
  epoch: 20,
  gpu: Integer(ENV['RED_CHAINER_GPU'] || -1),
  resume: nil,
  unit: 1000,
  out: 'result'
}

opt = OptionParser.new
opt.on('-b', '--batchsize VALUE', "Number of images in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on('-e', '--epoch VALUE', "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on('-g', '--gpu VALUE', "GPU ID (negative value indicates CPU) (default: #{args[:gpu]})") { |v| args[:gpu] = v.to_i }
opt.on('-f', '--frequency VALUE', "Frequency of taking a snapshot (default: #{args[:frequency]})") { |v| args[:frequency] = v.to_i }
opt.on('-o', '--out VALUE', "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on('-r', '--resume VALUE', "Resume the training from snapshot") { |v| args[:resume] = v }
opt.on('-u', '--unit VALUE', "Number of units (default: #{args[:unit]})") { |v| args[:unit] = v.to_i }
opt.parse!(ARGV)

puts "GPU: #{args[:gpu]}"
puts "# unit: #{args[:unit]}"
puts "# Minibatch-size: #{args[:batchsize]}"
puts "# epoch: #{args[:epoch]}"
puts

device = Chainer::Device.create(args[:gpu])
Chainer::Device.change_default(device)

lossfun = -> (x, t) { Chainer::Functions::Loss::SoftmaxCrossEntropy.new(ignore_label: nil).(x, t) }
model = Chainer::Links::Model::Classifier.new(MLP.new(args[:unit], 10), lossfun)

optimizer = Chainer::Optimizers::Adam.new
optimizer.setup(model)
train, test = Chainer::Datasets::MNIST.get_mnist

train_iter = Chainer::Iterators::SerialIterator.new(train, args[:batchsize])
test_iter = Chainer::Iterators::SerialIterator.new(test, args[:batchsize], repeat: false, shuffle: false)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: device)
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

trainer.extend(Chainer::Training::Extensions::Evaluator.new(test_iter, model, device: args[:gpu]))

# Take a snapshot for each specified epoch
frequency = args[:frequency] == -1 ? args[:epoch] : [1, args[:frequency]].max
trainer.extend(Chainer::Training::Extensions::Snapshot.new, trigger: [frequency, 'epoch'], priority: -100)

trainer.extend(Chainer::Training::Extensions::LogReport.new)
trainer.extend(Chainer::Training::Extensions::PrintReport.new(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(Chainer::Training::Extensions::ProgressBar.new)

if args[:resume]
  Chainer::Serializers::MarshalDeserializer.load_file(args[:resume], trainer)
end

trainer.run
