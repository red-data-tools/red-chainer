# frozen_string_literal: true

require 'chainer/functions/loss/mean_squared_error'

class Chainer::Functions::Loss::MeanSquaredErrorTest < Test::Unit::TestCase

  def _setup()
    @x0 = Numo::SFloat.new([4, 3]).rand(2) - 1
    @x1 = Numo::SFloat.new([4, 3]).rand(2) - 1
  end

  def check_forward(x0_data, x1_data)
    x0 = Chainer::Variable.new(x0_data)
    x1 = Chainer::Variable.new(x1_data)
    loss = Chainer::Functions::Loss::MeanSquaredError.mean_squared_error(x0, x1)
    loss_value = loss.data
    assert_equal(Numo::SFloat, loss_value.class)
    assert_equal([], loss_value.shape)
    loss_expect = 0.0
    @x0.each_with_index{|x,*i| loss_expect += (@x0[*i] - @x1[*i]) ** 2}
    loss_expect = (loss_expect)/(@x0.size).to_f
    assert_equal(loss_expect, loss_value)
  end

  def test_forward_cpu()
    _setup()
    check_forward(@x0, @x1)
  end
end
