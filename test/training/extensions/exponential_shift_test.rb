class Chainer::Training::Extensions::ExponentialShiftTest < Test::Unit::TestCase
  data({
    test1: {
      case: { rate: 0.5, options: {} },
      expected: 0.005
    },
    test2: {
      case: { rate: 0.4, options: { init: 2.0 } },
      expected: 0.8
    }
  })
  def test_exponential_shift(data)
    model = Chainer::Links::Model::Classifier.new(1)
    optimizer = Chainer::Optimizers::MomentumSGD.new
    optimizer.setup(model)
    train_iter = Chainer::Iterators::SerialIterator.new(Numo::DFloat[1, 2, 3], 1)
    updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer)
    trainer = Chainer::Training::Trainer.new(updater)
    extension = Chainer::Training::Extensions::ExponentialShift.new("lr", data[:case][:rate], **data[:case][:options])
    extension.init(trainer)
    extension.(trainer)

    assert_equal(data[:expected], optimizer.lr)
  end
end
