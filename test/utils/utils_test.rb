# frozen_string_literal: true

require 'chainer/utils/array'

class Chainer::Utils::ArrayTest < Test::Unit::TestCase
  data = {
    'test1' => {dtype: nil},
    'test2' => {dtype: Numo::SFloat},
    'test3' => {dtype: Numo::DFloat}}

  def _setup(data)
    @dtype = data[:dtype]
  end

  data(data)
  def test_scalar(data)
    _setup(data)
    x = Chainer::Utils::Array.force_array(Numo::SFloat.cast(1), dtype=@dtype)
    assert_true(x.is_a? Numo::NArray)
    if @dtype.nil?
      assert_equal(Numo::SFloat, x.class)
    else
      assert_equal(@dtype, x.class)
    end
  end

  data(data)
  def test_0dim_array(data)
    _setup(data)
    x = Chainer::Utils::Array.force_array(Numo::SFloat.cast(1), dtype=@dtype)
    assert_true(x.is_a? Numo::NArray)
    if @dtype.nil?
      assert_equal(Numo::SFloat, x.class)
    else
      assert_equal(@dtype, x.class)
    end
  end

  data(data)
  def test_array(data)
    _setup(data)
    x = Chainer::Utils::Array.force_array(Numo::SFloat.cast([1]), dtype=@dtype)
    assert_true(x.is_a? Numo::NArray)
    if @dtype.nil?
      assert_equal(Numo::SFloat, x.class)
    else
      assert_equal(@dtype, x.class)
    end
  end
end
