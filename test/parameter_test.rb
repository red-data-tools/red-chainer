# frozen_string_literal: true

require 'chainer'

class TestUninitializedParameter < Test::Unit::TestCase
  def setup
    @a = xm::SFloat.new(3, 2).rand
    @b = @a.class.new(*@a.shape).rand
  end

  def test_initialize_node
    initializer = Chainer::Initializers::Normal.new(dtype: xm::DFloat)
    x = Chainer::Parameter.new(initializer: initializer)
    x.init([2, 3])
    assert_equal([2, 3], x.node.shape)
    assert_equal(xm::DFloat, x.node.dtype)
  end
end
