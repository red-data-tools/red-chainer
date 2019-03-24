# frozen_string_literal: true

require 'chainer'

class SimpleLink < Chainer::Link
  attr_reader :param

  def initialize(w, g)
    super()
    init_scope do
      @param = Chainer::Parameter.new(initializer: w)
      @param.grad = g
    end
  end
end

class Chainer::WeightDecayTest < Test::Unit::TestCase
  def setup
    @target = SimpleLink.new(
      xm::SFloat.new(2, 3).seq,
      xm::SFloat.new(6).seq(-2).reverse.reshape(2, 3)
    )
  end

  def check_weight_decay
    w = @target.param.data
    g = @target.param.grad

    decay = 0.2
    expect = w - g - decay * w

    opt = Chainer::Optimizers::MomentumSGD.new(lr: 1)
    opt.setup(@target)
    opt.add_hook(Chainer::WeightDecay.new(decay))
    opt.update()
    Chainer::Testing.assert_allclose(expect, @target.param.data)
  end

  def test_weight_decay
    check_weight_decay
  end
end
