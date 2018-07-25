# frozen_string_literal: true

require 'chainer/utils/array'

class Chainer::Utils::ArrayTest < Test::Unit::TestCase
  sub_test_case "force_array" do
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

  sub_test_case "take" do
    data({
      axis0: {
        indices: 1,
        axis: 0,
        expect: Numo::DFloat[[12, 13, 14, 15],
                             [16, 17, 18, 19],
                             [20, 21, 22, 23]]
      },
      axis1: {
        indices: 1,
        axis: 1,
        expect: Numo::DFloat[[ 4,  5,  6,  7],
                             [16, 17, 18, 19]]
      },
      axis2: {
        indices: 1,
        axis: 2,
        expect: Numo::DFloat[[ 1,  5,  9],
                             [13, 17, 21]]
      },
      axis1_array: {
        indices: [1],
        axis: 1,
        expect: Numo::DFloat[[[ 4,  5,  6,  7]],
                             [[16, 17, 18, 19]]]
      },
      axis1_array12: {
        indices: [1, 2],
        axis: 1,
        expect: Numo::DFloat[[[ 4,  5,  6,  7],
                              [ 8,  9, 10, 11]],
                             [[16, 17, 18, 19],
                              [20, 21, 22, 23]]]
      },
      axis2_array12: {
        indices: [1, 2],
        axis: 2,
        expect: Numo::DFloat[[[ 1,  2],
                              [ 5,  6],
                              [ 9, 10]],
                             [[13, 14],
                              [17, 18],
                              [21, 22]]]
      }
    })
    def test_take(data)
      x = Numo::DFloat.new(2,3,4).seq
      result = Chainer::Utils::Array.take(x, data[:indices], axis: data[:axis])
      assert_equal data[:expect], result
    end
  end
end
