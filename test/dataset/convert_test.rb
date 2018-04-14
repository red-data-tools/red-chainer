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
    for i in arrays.size.times
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
