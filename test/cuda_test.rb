# frozen_string_literal: true

require 'chainer'

class TestCuda < Test::Unit::TestCase
  def test_get_array_module_for_numpy()
    assert_equal(Numo::NArray, Chainer::get_array_module(Numo::NArray[]))
    assert_equal(Numo::NArray, Chainer::get_array_module(Chainer::Variable.new(Numo::NArray[])))
  end
end
