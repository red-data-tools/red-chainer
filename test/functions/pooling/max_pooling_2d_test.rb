class Chainer::Functions::Pooling::MaxPooling2DTest < Test::Unit::TestCase
  data(
    test1: {
      case: {
        x: xm::DFloat.new(1, 3, 4, 6).seq,
        ksize: 2,
        options: {}
      },
      expected: xm::DFloat[[[[7.0, 9.0, 11.0], [19.0, 21.0, 23.0]], [[31.0, 33.0, 35.0], [43.0, 45.0, 47.0]], [[55.0, 57.0, 59.0], [67.0, 69.0, 71.0]]]]
    },
    test2: {
      case: {
        x: xm::DFloat.new(1, 3, 4, 4).seq,
        ksize: 2,
        options: { stride: 2 }
      },
      expected: xm::DFloat[[[[5.0, 7.0], [13.0, 15.0]], [[21.0, 23.0], [29.0, 31.0]], [[37.0, 39.0], [45.0, 47.0]]]]
    },
    test3: {
      case: {
        x: xm::DFloat.new(1, 3, 4, 4).seq,
        ksize: 4,
        options: {}
      },
      expected: xm::DFloat[[[[15.0]], [[31.0]], [[47.0]]]]
    },
  )
  def test_max_pooling_2d(data)
    test_case = data[:case]
    actual = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(test_case[:x], test_case[:ksize], **test_case[:options])
    assert_equal(data[:expected], actual.data)
  end

  data({
    test1: {
      case: {
        x: xm::DFloat.new(2, 3, 6, 3).seq,
        gy: [xm::DFloat.new(2, 3, 4, 2).seq],
        ksize: 3,
        stride: 2,
        pad: 1,
        cover_all: true
      },
      expected: xm::DFloat[[[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 10.0, 12.0]], [[0.0, 0.0, 0.0], [0.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 10.0, 11.0], [0.0, 0.0, 0.0], [0.0, 26.0, 28.0]], [[0.0, 0.0, 0.0], [0.0, 16.0, 17.0], [0.0, 0.0, 0.0], [0.0, 18.0, 19.0], [0.0, 0.0, 0.0], [0.0, 42.0, 44.0]]], [[[0.0, 0.0, 0.0], [0.0, 24.0, 25.0], [0.0, 0.0, 0.0], [0.0, 26.0, 27.0], [0.0, 0.0, 0.0], [0.0, 58.0, 60.0]], [[0.0, 0.0, 0.0], [0.0, 32.0, 33.0], [0.0, 0.0, 0.0], [0.0, 34.0, 35.0], [0.0, 0.0, 0.0], [0.0, 74.0, 76.0]], [[0.0, 0.0, 0.0], [0.0, 40.0, 41.0], [0.0, 0.0, 0.0], [0.0, 42.0, 43.0], [0.0, 0.0, 0.0], [0.0, 90.0, 92.0]]]]
    },
    test2: {
      case: {
        x: xm::DFloat.new(2, 3, 4, 3).seq,
        gy: [xm::DFloat.new(2, 3, 3, 2).seq],
        ksize: 3,
        stride: 2,
        pad: 1,
        cover_all: true
      },
      expected: xm::DFloat[[[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 6.0, 8.0]], [[0.0, 0.0, 0.0], [0.0, 6.0, 7.0], [0.0, 0.0, 0.0], [0.0, 18.0, 20.0]], [[0.0, 0.0, 0.0], [0.0, 12.0, 13.0], [0.0, 0.0, 0.0], [0.0, 30.0, 32.0]]], [[[0.0, 0.0, 0.0], [0.0, 18.0, 19.0], [0.0, 0.0, 0.0], [0.0, 42.0, 44.0]], [[0.0, 0.0, 0.0], [0.0, 24.0, 25.0], [0.0, 0.0, 0.0], [0.0, 54.0, 56.0]], [[0.0, 0.0, 0.0], [0.0, 30.0, 31.0], [0.0, 0.0, 0.0], [0.0, 66.0, 68.0]]]]
    },
    test3: {
      case: {
        x: xm::DFloat.new(1, 2, 2, 2).seq,
        gy: [xm::DFloat.new(1, 2, 4, 4).seq],
        ksize: 1,
        stride: 1,
        pad: 1,
        cover_all: true
      },
      expected: xm::DFloat[[[[5.0, 6.0], [9.0, 10.0]], [[21.0, 22.0], [25.0, 26.0]]]]
    },
  }) 
  def test_backward(data)
    c = data[:case]
    pooling = Chainer::Functions::Pooling::MaxPooling2D.new(c[:ksize], stride: c[:stride], pad: c[:pad], cover_all: c[:cover_all])
    pooling.(c[:x])
    gy = pooling.backward(c[:x], c[:gy])
    assert_equal(data[:expected], gy[0])
  end
end
