# frozen_string_literal: true

class Chainer::Optimizers::MomentumSGDTest < Test::Unit::TestCase
  data({
    test1: {
      case: { lr: nil, momentum: nil },
      expected: xm::DFloat[0.96, 1.95, 2.94]
    },
    test2: {
      case: { lr: 0.05, momentum: 0.5 },
      expected: xm::DFloat[0.8 , 1.75, 2.7]
    }
  })
  def test_momentum_sgd(data)
    var = Chainer::Variable.new(xm::DFloat[1, 2, 3])
    var.grad = xm::DFloat[4, 5, 6]
    sgd = Chainer::Optimizers::MomentumSGD.new(lr: data[:case][:lr], momentum: data[:case][:momentum])
    opt = sgd.create_update_rule
    opt.instance_variable_set(:@state, {})
    opt.init_state(var)
    opt.update_core(var)
    assert_equal(data[:expected], var.data)
  end
end
