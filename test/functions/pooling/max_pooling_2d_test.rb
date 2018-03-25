class Chainer::Functions::Pooling::MaxPooling2DTest < Test::Unit::TestCase
  data(
    test1: {
      case: {
        x: Numo::DFloat.new(1, 3, 4, 6).seq,
        ksize: 2,
        options: {}
      },
      expected: Numo::DFloat[[[[7.0, 9.0, 11.0], [19.0, 21.0, 23.0]], [[31.0, 33.0, 35.0], [43.0, 45.0, 47.0]], [[55.0, 57.0, 59.0], [67.0, 69.0, 71.0]]]]
    },
    test2: {
      case: {
        x: Numo::DFloat.new(1, 3, 4, 4).seq,
        ksize: 2,
        options: { stride: 2 }
      },
      expected: Numo::DFloat[[[[5.0, 7.0], [13.0, 15.0]], [[21.0, 23.0], [29.0, 31.0]], [[37.0, 39.0], [45.0, 47.0]]]]
    },
    test3: {
      case: {
        x: Numo::DFloat.new(1, 3, 4, 4).seq,
        ksize: 4,
        options: {}
      },
      expected: Numo::DFloat[[[[15.0]], [[31.0]], [[47.0]]]]
    },
  )
  def test_max_pooling_2d(data)
    test_case = data[:case]
    actual = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(test_case[:x], test_case[:ksize], **test_case[:options])
    assert_equal(data[:expected], actual.data)
  end
end
