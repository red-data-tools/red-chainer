require "chainer"
require "optionparser"

class RNNForLM < Chainer::Chain
  def initialize(n_vocab, n_units)
    super()
    init_scope do
      @embed = Chainer::Links::Connection::EmbedID.new(n_vocab, n_units)
      @l1 = Chainer::Links::Connection::LSTM.new(n_units, out_size: n_units)
      @l2 = Chainer::Links::Connection::LSTM.new(n_units, out_size: n_units)
      @l3 = Chainer::Links::Connection::Linear.new(n_units, out_size: n_vocab)
    end

    params do |param|
      param.data[*([:*] * param.data.shape.size)] = param.data.class.new(param.data.shape).rand(-0.1, 0.1)
    end
  end

  def reset_state
    @l1.reset_state()
    @l2.reset_state()
  end

  def call(x)
    h0 = @embed.(x)
    h1 = @l1.(Chainer::Functions::Noise::Dropout.dropout(h0))
    h2 = @l2.(Chainer::Functions::Noise::Dropout.dropout(h1))
    @l3.(Chainer::Functions::Noise::Dropout.dropout(h2))
  end
end

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator < Chainer::Dataset::Iterator
  attr_reader :is_new_epoch, :epoch

  def initialize(dataset, batch_size, repeat: true)
    @dataset = dataset
    @batch_size = batch_size  # batch size
    # Number of completed sweeps over the dataset. In this case, it is
    # incremented if every word is visited at least once after the last
    # increment.
    @epoch = 0
    # True if the epoch is incremented at the last iteration.
    @is_new_epoch = false
    @repeat = repeat
    # Offsets maintain the position of each sequence in the mini-batch.
    @offsets = batch_size.times.map {|i| i * length / batch_size }
    # NOTE: this is not a count of parameter updates. It is just a count of
    # calls of ``__next__``.
    @iteration = 0
  end

  # This iterator returns a list representing a mini-batch. Each item
  # indicates a different position in the original sequence. Each item is
  # represented by a pair of two word IDs. The first word is at the
  # "current" position, while the second word at the next position.
  # At each iteration, the iteration count is incremented, which pushes
  # forward the "current" position.
  def next
    # If not self.repeat, this iterator stops at the end of the first
    # epoch (i.e., when all words are visited once).
    if !@repeat && @iteration * @batch_size >= length
      raise StopIteration.new
    end
    cur_words = get_words
    @iteration += 1
    next_words = get_words

    epoch = @iteration * @batch_size / length
    @is_new_epoch = @epoch < epoch
    @epoch = epoch if @is_new_epoch

    pp cur_words
    pp next_words
    cur_words.zip(next_words)
  end

  def length
    @dataset.size
  end

  def epoch_detail
    # Floating point version of epoch.
    @iteration * @batch_size / length
  end

  def get_words
    # It returns a list of current words.
    @offsets.map {|offset| @dataset[(offset + @iteration) % length]}
  end

  def serialize(serializer)
    # It is important to serialize the state to be recovered on resume.
    @iteration = serializer('iteration', @iteration)
    @epoch = serializer('epoch', @epoch)
  end
end

# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater < Chainer::Training::StandardUpdater

  def initialize(train_iter, optimizer, bprop_len, device = -1)
    super(train_iter, optimizer, device: device)
    @bprop_len = bprop_len
  end

  # The core part of the update routine can be customized by overriding.
  def update_core
    loss = 0
    # When we pass one iterator and optimizer to StandardUpdater.__init__,
    # they are automatically named 'main'.
    train_iter = get_iterator(:main)
    optimizer = get_optimizer(:main)

    # Progress the dataset iterator for bprop_len words at each iteration.
    @bprop_len.times do |i|
      # Get the next batch (a list of tuples of two word IDs)
      batch = train_iter.next()

      # Concatenate the word IDs to matrices and send them to the device
      # self.converter does this job
      # (it is chainer.dataset.concat_examples by default)
      x, t = @converter.(batch, device: @device)

      # Compute the loss at this time step and accumulate it
      loss += optimizer.target.(Chainer::Variable.new(x), Chainer::Variable.new(t))
    end

    optimizer.target.cleargrads()  # Clear the parameter gradients
    loss.backward()  # Backprop
    loss.unchain_backward()  # Truncate the graph
    optimizer.update()  # Update the parameters
  end
end

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
compute_perplexity = Proc.new do |result|
  result['perplexity'] = np.exp(result['main/loss'])
  if result.key?('validation/main/loss')
    result['val_perplexity'] = Numo::NMath.exp(result['validation/main/loss'])
  end
end


args = {
  unitsize: 650,
  batchsize: 20,
  bproplen: 35,
  epoch: 39,
  gradclip: 5,
  resume: nil,
  gpu: Integer(ENV['RED_CHAINER_GPU'] || -1),
  model: "model.npz",
  negativesize: 5,
  out: 'result',
  test: false,
}

opt = OptionParser.new
opt.on("-u", "--unitsize VALUE", "Number of units (default: #{args[:unitsize]})") { |v| args[:unitsize] = v.to_i }
opt.on("-b", "--batchsize VALUE", "Number of words in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on("-l", "--bproplene VALUE", "Number of words in each mini-batch(= length of truncated BPTT)  (default: #{args[:bproplen]})") { |v| args[:bproplen] = v.to_i }
opt.on('-g', '--gpu VALUE', "GPU ID (negative value indicates CPU) (default: #{args[:gpu]})") { |v| args[:gpu] = v.to_i }
opt.on("-e", "--epoch VALUE", "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on("-c", "--gradclip VALUE", "Gradient norm threshold to clip (default: #{args[:gradclip]})") { |v| args[:gradclip] = v.to_i }
opt.on("-r", "--resume VALUE", "Resume the training from snapshot") { |v| args[:resume] = v }
opt.on("-m", "--model VALUE", "The model to use: skipgram or cbow (default: #{args[:model]})") { |v| args[:model] = v }
opt.on("--negativesize VALUE", "Number of negative sample (default: #{args[:negativesize]})") { |v| args[:negativesize] = v.to_f }
opt.on("-o", "--out VALUE", "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on("--test", "Use tiny datasets for quick tests") { args[:test] = true }
opt.parse!(ARGV)

puts "GPU: #{args[:gpu]}"

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
# Load the Penn Tree Bank long word sequence dataset

device = Chainer::Device.create(args[:gpu])
Chainer::Device.change_default(device)

train = Datasets::PennTreebank.new(type: :train)
valid = Datasets::PennTreebank.new(type: :valid)
test = Datasets::PennTreebank.new(type: :test)

bow = BagOfWords.new(:train, train)
bow.add(:valid, valid)
bow.add(:test, test)
n_vocab =  bow.vocabularies.size + 1
puts "#vocab = #{n_vocab}"

if args[:test]
  train_ids = bow.ids[:train].take(100)
  valid_ids = bow.ids[:valid].take(100)
  test_ids = bow.ids[:valid].take(100)
else
  train_ids = bow.ids[:train]
  valid_ids = bow.ids[:valid]
  test_ids = bow.ids[:test]
end

train_iter = ParallelSequentialIterator.new(train_ids, args[:batchsize])
val_iter = ParallelSequentialIterator.new(valid_ids, 1, repeat: false)
test_iter = ParallelSequentialIterator.new(test_ids, 1, repeat: false)

# Prepare an RNNLM model
rnn = RNNForLM.new(n_vocab, args[:unitsize])
model = Chainer::Links::Model::Classifier.new(rnn)
model.compute_accuracy = false  # we only want the perplexity

# Set up an optimizer
optimizer = Chainer::Optimizers::SGD.new(lr: 1.0)
optimizer.setup(model)
optimizer.add_hook(Chainer::GradientClipping.new(args[:gradclip]))

# Set up a trainer
updater = BPTTUpdater.new(train_iter, optimizer, args[:bproplen], args[:gpu])
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

eval_model = model.dup  # Model with shared params and distinct states
eval_rnn = eval_model.predictor
trainer.extend(Chainer::Training::Extensions::Evaluator.new(
    val_iter,
    eval_model,
    device: args[:gpu],
    # Reset the RNN state at the beginning of each evaluation
    eval_hook: -> (_) { eval_rnn.reset_state }
))

interval = args[:test] ? 10 : 500
trainer.extend(Chainer::Training::Extensions::LogReport.new(postprocess: compute_perplexity, trigger: [interval, 'iteration']))
trainer.extend(
  Chainer::Training::Extensions::PrintReport.new(['epoch', 'iteration', 'perplexity', 'val_perplexity']),
  trigger: [interval, 'iteration']
)
trainer.extend(Chainer::Training::Extensions::ProgressBar.new(update_interval: args[:test] ? 1 : 10))
# trainer.extend(Chainer::Training::Extensions::Snapshot.snapshot)
# trainer.extend(Chainer::Training::Extensions::Snapshot.snapshot_object(target: model, save_class: 'model_iter_{.updater.iteration}'))
# if args[:resume]
#   Chainer::Serializers::MarshalDeserializer.load_file(args[:resume], trainer)
# end

trainer.run()

# Evaluate the final model
print('test')
eval_rnn.reset_state()
evaluator = Chainer::Extensions::Evaluator.new(test_iter, eval_model, device: args[:gpu])
result = evaluator.()
puts "test perplexity: #{Numo::NMath.exp(result['main/loss'].to_f)}"

# Serialize the final model
#chainer.serializers.save_npz(args.model, model)
