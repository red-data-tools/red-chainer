# frozen_string_literal: true

require 'chainer'

class TestBackend < Test::Unit::TestCase
  def test_get_array_module_for_numo()
    assert Numo == Chainer.get_array_module(Numo::NArray[])
    assert Numo == Chainer.get_array_module(Chainer::Variable.new(Numo::NArray[]))
  end

  def test_get_array_module_for_cumo()
    assert_equal(Cumo, Chainer.get_array_module(Cumo::NArray[]))
    assert_equal(Cumo, Chainer.get_array_module(Chainer::Variable.new(Cumo::NArray[])))
  end if Chainer::CUDA.available?

  def test_array_p_for_numo()
    assert_equal(Numo, Chainer.get_array_module(Numo::NArray[]))
    assert_equal(Numo, Chainer.get_array_module(Chainer::Variable.new(Numo::NArray[])))
  end

  def test_array_p_for_cumo()
    assert_equal(Cumo, Chainer.get_array_module(Cumo::NArray[]))
    assert_equal(Cumo, Chainer.get_array_module(Chainer::Variable.new(Cumo::NArray[])))
  end if Chainer::CUDA.available?
end
