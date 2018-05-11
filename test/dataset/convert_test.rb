# frozen_string_literal: true

require 'chainer/dataset/convert'
require 'chainer/testing/array'

class TestConcatExamples < Test::Unit::TestCase
  def get_arrays_to_concat(xumo)
    return 5.times.map{|_| xumo::DFloat.new(2, 3).rand()}
  end

  def check_device(array, device)
    if device
      # T.B.I (GPU Check)
    end
  end

  def check_concat_arrays(arrays, device: nil)
    array = Chainer::Dataset::Convert.method(:concat_examples).call(arrays, device: device)
    assert_equal([arrays.size] + arrays[0].shape, array.shape)
    check_device(array, device)
    array.to_a.zip(arrays.to_a).each do |x, y|
      assert_true array.class.cast(x) == array.class.cast(y)
    end
  end

  def test_concat_arrays_cpu()
    arrays = get_arrays_to_concat(Numo)
    check_concat_arrays(arrays)
  end

  def get_tuple_arrays_to_concat(xumo)
    return 5.times.map{|_| [xumo::DFloat.new(2, 3).rand(), xumo::DFloat.new(3, 4).rand()]}
  end

  def check_concat_tuples(tuples, device: nil)
    arrays = Chainer::Dataset::Convert.method(:concat_examples).call(tuples, device: device)
    assert_equal(tuples[0].size, arrays.size)
    arrays.size.times do |i|
      shape = [tuples.size] + tuples[0][i].shape
      assert_equal(shape, arrays[i].shape)
      check_device(arrays[i], device)
      arrays[i].to_a.zip(tuples.to_a).each do |x, y|
        assert_true arrays[i].class.cast(x) == arrays[i].class.cast(y[i])
      end
    end
  end

  def test_concat_tuples_cpu()
    tuples = get_tuple_arrays_to_concat(Numo)
    check_concat_tuples(tuples)
  end
end

class TestConcatExamplesWithPadding < Test::Unit::TestCase
  def check_concat_arrays_padding(xumo)
    arrays = [xumo::DFloat.new(3, 4).rand(), xumo::DFloat.new(2, 5).rand(), xumo::DFloat.new(4, 3).rand()]
    array = Chainer::Dataset::Convert.method(:concat_examples).call(arrays, padding: 0)

    assert_equal([3, 4, 5], array.shape)
    assert_equal(arrays[0].class, array.class)
    arrays = arrays.map{|a| array.class.cast(a)}
    assert_true array[0, 0...3, 0...4].nearly_eq(arrays[0]).all?
    assert_true array[0, 3..-1, 0..-1].nearly_eq(0).all?
    assert_true array[0, 0..-1, 4..-1].nearly_eq(0).all?
    assert_true array[1, 0...2, 0...5].nearly_eq(arrays[1]).all?
    assert_true array[1, 2..-1, 0..-1].nearly_eq(0).all?
    assert_true array[2, 0...4, 0...3].nearly_eq(arrays[2]).all?
    assert_true array[2, 0..-1, 3..-1].nearly_eq(0).all?
  end

  def test_concat_arrays_padding_cpu()
    check_concat_arrays_padding(Numo)
  end

  def check_concat_tuples_padding(xumo)
    tuples = [[xumo::DFloat.new(3, 4).rand(), xumo::DFloat.new(2, 5).rand()],
              [xumo::DFloat.new(4, 4).rand(), xumo::DFloat.new(3, 4).rand()],
              [xumo::DFloat.new(2, 5).rand(), xumo::DFloat.new(2, 6).rand()]]
    arrays = Chainer::Dataset::Convert.method(:concat_examples).call(tuples, padding: 0)

    assert_equal(2, arrays.size)
    assert_equal([3, 4, 5], arrays[0].shape)
    assert_equal([3, 3, 6], arrays[1].shape)
    assert_equal(tuples[0][0].class, arrays[0].class)
    assert_equal(tuples[0][1].class, arrays[1].class)
    tuples.size.times do |i|
      tuples[i] = [tuples[i][0], tuples[i][1]]
    end

    arrays = arrays.to_a
    assert_true arrays[0][0, 0...3, 0...4].nearly_eq(tuples[0][0]).all?
    assert_true arrays[0][0, 3..-1, 0..-1].nearly_eq(0).all?
    assert_true arrays[0][0, 0..-1, 4..-1].nearly_eq(0).all?
    assert_true arrays[0][1, 0...4, 0...4].nearly_eq(tuples[1][0]).all?
    assert_true arrays[0][1, 0..-1, 4..-1].nearly_eq(0).all?
    assert_true arrays[0][2, 0...2, 0...5].nearly_eq(tuples[2][0]).all?
    assert_true arrays[0][2, 2..-1, 0..-1].nearly_eq(0).all?
    assert_true arrays[1][0, 0...2, 0...5].nearly_eq(tuples[0][1]).all?
    assert_true arrays[1][0, 2..-1, 0..-1].nearly_eq(0).all?
    assert_true arrays[1][0, 0..-1, 5..-1].nearly_eq(0).all?
    assert_true arrays[1][1, 0...3, 0...4].nearly_eq(tuples[1][1]).all?
    #assert_true arrays[1][1, 3..-1, 0..-1].nearly_eq(0).all? # range error
    assert_true arrays[1][1, 0..-1, 4..-1].nearly_eq(0).all?
    assert_true arrays[1][2, 0...2, 0...6].nearly_eq(tuples[2][1]).all?
    assert_true arrays[1][2, 2..-1, 0..-1].nearly_eq(0).all?
  end

  def test_concat_tuples_padding_cpu()
    check_concat_tuples_padding(Numo)
  end
end

class TestConcatExamplesWithBuiltInTypes < Test::Unit::TestCase
  data = {
    'test1' => {padding: nil},
    'test2' => {padding: 0}}

  @@int_arrays = [1, 2, 3]
  @@float_arrays = [1.0, 2.0, 3.0]

  def check_device(array, device)
    if device && device >= 0
      # T.B.I (GPU Check)
    else
      assert_true array.is_a?(Numo::NArray)
    end
  end

  def check_concat_arrays(arrays, device:, expected_type:)
    array = Chainer::Dataset::Convert.method(:concat_examples).call(arrays, device: device, padding: @padding)
    assert_equal([arrays.size], array.shape)
    check_device(array, device)

    array.to_a.zip(arrays.to_a).each do |x, y|
      assert_true Numo::NArray.cast(y).nearly_eq(Numo::NArray.cast(x)).all?
    end
  end

  data(data)
  def test_concat_arrays_cpu(data)
    @padding = data[:padding]

    [-1, nil].each do |device|
      check_concat_arrays(@@int_arrays, device: device, expected_type: Numo::Int64)
      check_concat_arrays(@@float_arrays, device: device, expected_type: Numo::DFloat)
    end
  end
end
