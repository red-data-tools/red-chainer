# frozen_string_literal: true

require 'chainer'

class TestCuda < Test::Unit::TestCase
  def test_get_array_module_for_numo()
    assert_equal(Numo, Chainer.get_array_module(Numo::NArray[]))
    assert_equal(Numo, Chainer.get_array_module(Chainer::Variable.new(Numo::NArray[])))
  end

  if Chainer::CUDA.available?
    def test_get_array_module_for_cumo()
      assert_equal(Cumo, Chainer.get_array_module(Cumo::NArray[]))
      assert_equal(Cumo, Chainer.get_array_module(Chainer::Variable.new(Cumo::NArray[])))
    end
  end
end
