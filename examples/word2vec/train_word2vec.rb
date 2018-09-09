require 'chainer'
require_relative 'models/skip-gram'
require_relative 'models/cbow'
require 'optparse'
require 'datasets'

args = {
  unitsize: 100,
  windowsize: 5,
  batchsize: 1000,
  epoch: 20,
  model: 'skipgram',
  negativesize: 5,
  out_type: 'original',
  out: 'result',
  test: false,
}

opt = OptionParser.new
opt.on('-u', '--unitsize VALUE', "Number of units (default: #{args[:unitsize]})") { |v| args[:unitsize] = v.to_i }
opt.on('-w', '--windowsize VALUE', "Window size (default: #{args[:windowsize]})") { |v| args[:windowsize] = v.to_i }
opt.on('-b', '--batchsize VALUE', "Number of words in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on('-e', '--epoch VALUE', "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on('-m', '--model VALUE', "The model to use: skipgram or cbow (default: #{args[:model]})") { |v| args[:model] = v }
opt.on('--negativesize VALUE', "Number of negative sample (default: #{args[:negativesize]})") { |v| args[:negativesize] = v.to_f }
# todo implement hsm, ns
# opt.on('-o', '--out-type VALUE', "Output model type: hsm or ns or original (default: #{args[:out_type]})") { |v| args[:out_type] = v }
opt.on('--out VALUE', "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on('--test') { args[:test] = true }
opt.parse!(ARGV)

if args[:model] == 'skipgram'
  puts 'Using Skip-Gram model'
  model_class = SkipGram
elsif args[:model] == 'cbow'
  puts 'Using Continuous Bag of Words model'
  model_class = CBoW
end

class SoftmaxCrossEntropyLoss < Chainer::Chain
  def initialize(n_in, n_out)
    super()
    init_scope do
      @out = Chainer::Links::Connection::Linear.new(n_in, out_size: n_out, initial_w: 0)
    end
  end

  def call(x, t)
      return Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@out.(x), t)
  end
end

class WindowIterator < Chainer::Dataset::Iterator
  attr_reader :epoch, :is_new_epoch

  def initialize(dataset, window, batch_size, repeat: true)
    @dataset = Numo::Int32[dataset]
    @window = window
    @batch_size = batch_size
    @repeat = repeat

    @order = Numo::Int32[(window...(@dataset.size - window)).to_a.shuffle]
    @current_position = 0
    @epoch = 0
    @is_new_epoch = false
  end

  def next
    raise StopIteration if !@repeat && @epoch > 0

    i = @current_position
    i_end = i + @batch_size
    position = @order[i...[i_end, @order.size].min]
    w = Numo::Int32.new(1).rand(1, @window)[0]
    seq = Numo::Int32.new(w).seq
    offset = Numo::NArray.concatenate([seq - w, seq + 1])
    pos = position.reshape(position.size, 1) + offset
    context = Chainer::Utils::Array.take(@dataset, pos)
    center = Chainer::Utils::Array.take(@dataset, position)

    if i_end >= @order.size
      @order = Numo::Int32[@order.to_a.shuffle]
      @epoch += 1
      @is_new_epoch = true
      @current_position = 0
    else
      @is_new_epoch = false
      @current_position = i_end
    end

    [center, context]
  end

  def epoch_detail
    @epoch + @current_position.to_f / @order.size
  end

  def serialize(serializer)
    @current_position = serializer.('current_position', @current_position)
    @epoch = serializer.('epoch', @epoch)
    @is_new_epoch = serializer('is_new_epoch', @is_new_epoch)
    serializer.('order', @order) if @order
  end
end

class BagOfWords
  attr_reader :counter, :vocabularies, :ids

  def initialize(name, enum)
    @ids = {}
    @counter = {}
    @vocabularies = {}
    add(name, enum)
  end

  def add(name, enum)
    ids = []
    @ids[name] = ids
    enum.each do |v|
      @counter[v.word] = @counter[v.word].yield_self{|c| c ? c + 1 : 0 }
      unless @vocabularies.key?(v.word)
        @vocabularies[v.word] = @vocabularies.size
      end
      ids << @vocabularies[v.word]
    end
  end

  def [](id)
    @vocabularies.keys[id]
  end
end

converter = Proc.new do |batch, device|
  [batch, device]
end

train = Datasets::PennTreebank.new(type: :train)
valid = Datasets::PennTreebank.new(type: :valid)

bow = BagOfWords.new(:train, train)
bow.add(:valid, valid)

n_vocab = bow.vocabularies.size + 1

if args[:test]
  train_ids = bow.ids[:train].take(100)
  valid_ids = bow.ids[:valid].take(100)
else
  train_ids = bow.ids[:train]
  valid_ids = bow.ids[:valid]
end

puts "n_vocab: #{n_vocab}"
puts "data length: #{train_ids.count}"

case args[:out_type]
when 'hsm'
  raise "Not impemented 'hsm'"
  # HSM = Chainer::Links::Connection::Linear.BinaryHierarchicalSoftmax
  # tree = HSM.create_huffman_tree(counts)
  # loss_func = HSM(args.unit, tree)
  # loss_func.W.data[...] = 0
when 'ns'
  raise "Not impemented 'ns'"
  # cs = [counts[w] for w in range(len(counts))]
  # loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
  # loss_func.W.data[...] = 0
when 'original'
  loss_func = SoftmaxCrossEntropyLoss.new(args[:unitsize], n_vocab)
else
  raise "Unknown output type: #{args[:out_type]}"
end

case args[:model]
when 'skipgram'
  model = SkipGram.new(n_vocab, args[:unitsize], loss_func)
when 'cbow'
  model = CBoW.new(n_vocab, args[:unitsize], loss_func)
else
  raise "Unknown model type: @{args[:model]}"
end

optimizer = Chainer::Optimizers::Adam.new
optimizer.setup(model)

train_iter = WindowIterator.new(train_ids, args[:windowsize], args[:batchsize])
valid_iter = WindowIterator.new(valid_ids, args[:windowsize], args[:batchsize], repeat: false)
updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, converter: converter, device: -1)
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

trainer.extend(Chainer::Training::Extensions::Evaluator.new(valid_iter, model, converter: converter, device: -1))
trainer.extend(Chainer::Training::Extensions::LogReport.new)
trainer.extend(Chainer::Training::Extensions::PrintReport.new(['epoch', 'main/loss', 'validation/main/loss']))
trainer.extend(Chainer::Training::Extensions::ProgressBar.new)
trainer.run()

open('word2vec.model', 'w') do |f|
  f.write("#{n_vocab} #{args[:unit]}\n")
  w = model.embed.w.data
  w.shape.first.times do |i|
    v = w[i, true].to_a.join(' ')
    f.write("#{bow[i]} #{v}\n")
  end
end
