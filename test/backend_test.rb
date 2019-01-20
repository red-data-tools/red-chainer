# frozen_string_literal: true

require 'chainer'

class TestBackend < Test::Unit::TestCase
  def test_get_array_module
    assert xm == Chainer.get_array_module(xm::NArray[])
    assert xm == Chainer.get_array_module(Chainer::Variable.new(xm::NArray[]))
  end

  def test_array_p
    assert_equal(xm, Chainer.get_array_module(xm::NArray[]))
    assert_equal(xm, Chainer.get_array_module(Chainer::Variable.new(xm::NArray[])))
  end
end
