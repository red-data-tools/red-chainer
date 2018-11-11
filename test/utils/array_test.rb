# frozen_string_literal: true

require 'chainer/utils/array'

class Chainer::Utils::ArrayTest < Test::Unit::TestCase
  sub_test_case "force_array" do
    data = {
      'test1' => {dtype: nil},
      'test2' => {dtype: xm::SFloat},
      'test3' => {dtype: xm::DFloat}}

    def _setup(data)
      @dtype = data[:dtype]
    end

    data(data)
    def test_scalar(data)
      _setup(data)
      x = Chainer::Utils::Array.force_array(xm::SFloat.cast(1), dtype=@dtype)
      assert_true(x.is_a? xm::NArray)
      if @dtype.nil?
        assert_equal(xm::SFloat, x.class)
      else
        assert_equal(@dtype, x.class)
      end
    end

    data(data)
    def test_0dim_array(data)
      _setup(data)
      x = Chainer::Utils::Array.force_array(xm::SFloat.cast(1), dtype=@dtype)
      assert_true(x.is_a? xm::NArray)
      if @dtype.nil?
        assert_equal(xm::SFloat, x.class)
      else
        assert_equal(@dtype, x.class)
      end
    end

    data(data)
    def test_array(data)
      _setup(data)
      x = Chainer::Utils::Array.force_array(xm::SFloat.cast([1]), dtype=@dtype)
      assert_true(x.is_a? xm::NArray)
      if @dtype.nil?
        assert_equal(xm::SFloat, x.class)
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
        expect: xm::DFloat[[12, 13, 14, 15],
                             [16, 17, 18, 19],
                             [20, 21, 22, 23]]
      },
      axis1: {
        indices: 1,
        axis: 1,
        expect: xm::DFloat[[ 4,  5,  6,  7],
                             [16, 17, 18, 19]]
      },
      axis2: {
        indices: 1,
        axis: 2,
        expect: xm::DFloat[[ 1,  5,  9],
                             [13, 17, 21]]
      },
      axis1_array: {
        indices: [1],
        axis: 1,
        expect: xm::DFloat[[[ 4,  5,  6,  7]],
                             [[16, 17, 18, 19]]]
      },
      axis1_array12: {
        indices: [1, 2],
        axis: 1,
        expect: xm::DFloat[[[ 4,  5,  6,  7],
                              [ 8,  9, 10, 11]],
                             [[16, 17, 18, 19],
                              [20, 21, 22, 23]]]
      },
      axis2_array12: {
        indices: [1, 2],
        axis: 2,
        expect: xm::DFloat[[[ 1,  2],
                              [ 5,  6],
                              [ 9, 10]],
                             [[13, 14],
                              [17, 18],
                              [21, 22]]]
      }
    })
    def test_take(data)
      x = xm::DFloat.new(2,3,4).seq
      result = Chainer::Utils::Array.take(x, data[:indices], axis: data[:axis])
      assert_equal data[:expect], result
    end
  end

  sub_test_case "rollaxis" do
    data({
      test1: {
        # shape : [3, 4, 5, 6] => [3, 6, 4, 5]
        a: xm::DFloat.ones([3, 4, 5, 6]),
        axis: 3,
        start: 1,
        expected: [3, 6, 4, 5]
      },
      test2: {
        # shape : [3, 4, 5, 6] => [5, 3, 4, 6]
        a: xm::DFloat.ones([3, 4, 5, 6]),
        axis: 2,
        start: 0,
        expected: [5, 3, 4, 6]
      },
      test3: {
        a: xm::DFloat.ones([3, 4, 5, 6]),
        axis: 1,
        start: 4,
        expected: [3, 5, 6, 4]
      },
      test4: {
        a: xm::DFloat.ones([2, 3, 2]),
        axis: 1,
        start: 0,
        expected: [3, 2, 2]

      }
    })

    def test_rollaxis(data)
       assert_equal(data[:expected], Chainer::Utils::Array.rollaxis(data[:a], data[:axis], start: data[:start]).shape)
    end
  end

  sub_test_case "broadcast_to" do
    data({
      test1: {
        # shape : [3] => [3]
        a: [1, 2, 3],
        shape: [3],
        expected: xm::SFloat[1, 2, 3]
      },
      test2: {
        # shape : [3] => [2, 3]
        a: [1, 2, 3],
        shape: [2, 3],
        expected: xm::SFloat[[1, 2, 3], [1, 2, 3]]
      },
      test3: {
        # shape : [3] => [2, 2, 3]
        a: [1, 2, 3],
        shape: [2, 2, 3],
        expected: xm::SFloat[[[1, 2, 3], [1, 2, 3]],
                               [[1, 2, 3], [1, 2, 3]]]
      },
      test4: {
        # shape : [2, 3] => [2, 3]
        a: [[0, 1, 2], [3, 4, 5]],
        shape: [2, 3],
        expected: xm::SFloat[[0, 1, 2], [3, 4, 5]]
      },
      test5: {
        # shape : [2, 3] => [2, 2, 3]
        a: [[0, 1, 2], [3, 4, 5]],
        shape: [2, 2, 3],
        expected: xm::SFloat[[[0, 1, 2], [3, 4, 5]],
                               [[0, 1, 2], [3, 4, 5]]]
      },
      test6: {
        # shape : [2, 3] => [2, 2, 2, 3]
        a: [[0, 1, 2], [3, 4, 5]],
        shape: [2, 2, 2, 3],
        expected: xm::SFloat[[[[0, 1, 2], [3, 4, 5]],
                                [[0, 1, 2], [3, 4, 5]]],
                               [[[0, 1, 2], [3, 4, 5]],
                                [[0, 1, 2], [3, 4, 5]]]]
      },
      test7: {
        # shape : [1, 3] => [1, 3]
        a: [[1, 2, 3]],
        shape: [1, 3],
        expected: xm::SFloat[[1, 2, 3]]
      },
      test8: {
        # shape : [1, 3] => [2, 3]
        a: [[1, 2, 3]],
        shape: [2, 3],
        expected: xm::SFloat[[1, 2, 3], [1, 2, 3]]
      },
      test9: {
        # shape : [1, 3] => [2, 2, 3]
        a: [[1, 2, 3]],
        shape: [2, 2, 3],
        expected: xm::SFloat[[[1, 2, 3], [1, 2, 3]],
                               [[1, 2, 3], [1, 2, 3]]]
      },
      test10: {
        # shape : [1, 3, 1] => [1, 3, 1]
        a: [[[1], [2], [3]]],
        shape: [1, 3, 1],
        expected: xm::SFloat[[[1], [2], [3]]]
      },
      test11: {
        # shape : [1, 3, 1] => [1, 3, 2]
        a: [[1], [2], [3]],
        shape: [1, 3, 2],
        expected: xm::SFloat[[[1, 1], [2, 2], [3, 3]]]
      },
      test12: {
        # shape : [1, 3, 1] => [2, 3, 1]
        a: [[1], [2], [3]],
        shape: [2, 3, 1],
        expected: xm::SFloat[[[1], [2], [3]], [[1], [2], [3]]]
      }
    })

    def test_broadcast_to(data)
      x = xm::SFloat.cast(data[:a])
      y = Chainer::Utils::Array.broadcast_to(x, data[:shape])
      assert_equal(data[:shape], y.shape)
      assert_equal(data[:expected], y)
    end
  end
end
