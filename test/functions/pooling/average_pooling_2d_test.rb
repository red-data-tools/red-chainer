class Chainer::Functions::Pooling::AveragePooling2DTest < Test::Unit::TestCase
  data(
    test1: {
      case: {
        x: Numo::SFloat.new(1, 3, 4, 6).seq,
        ksize: 2,
        options: {}
      },
      expected: Numo::SFloat[[[[ 3.5,  5.5,  7.5],
                               [15.5, 17.5, 19.5]],
                              [[27.5, 29.5, 31.5],
                               [39.5, 41.5, 43.5]],
                              [[51.5, 53.5, 55.5],
                               [63.5, 65.5, 67.5]]]]
    },
    test2: {
      case: {
        x: Numo::SFloat.new(1, 3, 4, 4).seq,
        ksize: 2,
        options: { stride: 2 }
      },
      expected: Numo::SFloat[[[[ 2.5,  4.5],
                               [10.5, 12.5]],
                              [[18.5, 20.5],
                               [26.5, 28.5]],
                              [[34.5, 36.5],
                               [42.5, 44.5]]]]
    },
    test3: {
      case: {
        x: Numo::SFloat.new(1, 3, 4, 4).seq,
        ksize: 4,
        options: { stride: 2, pad: 1 }
      },
      expected: Numo::SFloat[[[[ 2.8125,  3.375 ],
                               [ 5.0625,  5.625 ]],
                              [[11.8125, 12.375 ],
                               [14.0625, 14.625 ]],
                              [[20.8125, 21.375 ],
                               [23.0625, 23.625 ]]]]
    },
  )
  def test_average_pooling_2d(data)
    test_case = data[:case]
    actual = Chainer::Functions::Pooling::AveragePooling2D.average_pooling_2d(test_case[:x], test_case[:ksize], **test_case[:options])
    assert_equal(data[:expected], actual.data)
  end

  data({
    test1: {
      case: {
        x: Numo::SFloat.new(2, 3, 2, 2).seq,
        gy: [Numo::SFloat.new(2, 3, 1, 1).seq],
        ksize: 2,
        stride: 2,
        pad: 0,
        cover_all: false
      },
      expected: Numo::SFloat[[[[0.0 , 0.0  ],
                               [0.0 , 0.0  ]],
                              [[0.25, 0.25],
                               [0.25, 0.25]],
                              [[0.5 , 0.5 ],
                               [0.5 , 0.5 ]]],
                             [[[0.75, 0.75],
                               [0.75, 0.75]],
                              [[1.0 , 1.0  ],
                               [1.0 , 1.0  ]],
                              [[1.25, 1.25],
                               [1.25, 1.25]]]]
    },
    test2: {
      case: {
        x: Numo::SFloat.new(2, 2, 4, 4).seq,
        gy: [Numo::SFloat.new(2, 3 ,1, 1).seq],
        ksize: 6,
        stride: 8,
        pad: 1,
        cover_all: false
      },
      expected: Numo::SFloat[[[[0.0      , 0.0      , 0.0      , 0.0     ],
                               [0.0      , 0.0      , 0.0      , 0.0     ],
                               [0.0      , 0.0      , 0.0      , 0.0     ],
                               [0.0      , 0.0      , 0.0      , 0.0     ]],
                              [[0.0277778, 0.0277778, 0.0277778, 0.0277778],
                               [0.0277778, 0.0277778, 0.0277778, 0.0277778],
                               [0.0277778, 0.0277778, 0.0277778, 0.0277778],
                               [0.0277778, 0.0277778, 0.0277778, 0.0277778]],
                              [[0.0555556, 0.0555556, 0.0555556, 0.0555556],
                               [0.0555556, 0.0555556, 0.0555556, 0.0555556],
                               [0.0555556, 0.0555556, 0.0555556, 0.0555556],
                               [0.0555556, 0.0555556, 0.0555556, 0.0555556]]],
                             [[[0.0833333, 0.0833333, 0.0833333, 0.0833333],
                               [0.0833333, 0.0833333, 0.0833333, 0.0833333],
                               [0.0833333, 0.0833333, 0.0833333, 0.0833333],
                               [0.0833333, 0.0833333, 0.0833333, 0.0833333]],
                              [[0.1111111, 0.1111111, 0.1111111, 0.1111111],
                               [0.1111111, 0.1111111, 0.1111111, 0.1111111],
                               [0.1111111, 0.1111111, 0.1111111, 0.1111111],
                               [0.1111111, 0.1111111, 0.1111111, 0.1111111]],
                              [[0.1388889, 0.1388889, 0.1388889, 0.1388889 ],
                               [0.1388889, 0.1388889, 0.1388889, 0.1388889 ],
                               [0.1388889, 0.1388889, 0.1388889, 0.1388889 ],
                               [0.1388889, 0.1388889, 0.1388889, 0.1388889 ]]]]
    }
  })
  def test_backward(data)
    c = data[:case]
    pooling = Chainer::Functions::Pooling::AveragePooling2D.new(c[:ksize], stride: c[:stride], pad: c[:pad], cover_all: c[:cover_all])
    pooling.(c[:x])
    gy = pooling.backward_cpu(c[:x], c[:gy])
    d = 7
    assert_equal(round(data[:expected], 7), round(gy[0], 7))
  end

  def round(x, decimals)
    return nil if x.nil?
    t = 10 ** decimals
    (x * t).round / t
  end
end
