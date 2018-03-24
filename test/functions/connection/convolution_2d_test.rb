# frozen_string_literal: true

class Chainer::Functions::Connection::Convolution2DTest < Test::Unit::TestCase
  data({
    test1: {
      case: { x: Numo::DFloat.new(1, 1, 4, 4).seq, w: Numo::DFloat.new(2, 1, 3, 3).seq, options: {} },
      expected: Numo::DFloat[[[[258.0, 294.0], [402.0, 438.0]], [[663.0, 780.0], [1131.0, 1248.0]]]]
    },
    test2: {
      case: { x: Numo::DFloat.new(1, 2, 4, 4).seq, w: Numo::DFloat.new(2, 2, 3, 3).seq, options: {} },
      expected: Numo::DFloat[[[[2793.0, 2946.0], [3405.0, 3558.0]], [[7005.0, 7482.0], [8913.0, 9390.0]]]]
    },
    test3: {
      case: { x: Numo::DFloat.new(2, 2, 4, 4).seq, w: Numo::DFloat.new(2, 2, 3, 3).seq, options: {} },
      expected: Numo::DFloat[[[[2793.0, 2946.0], [3405.0, 3558.0]], [[7005.0, 7482.0], [8913.0, 9390.0]]], [[[7689.0, 7842.0], [8301.0, 8454.0]], [[22269.0, 22746.0], [24177.0, 24654.0]]]]
    },
    test4: {
      case: { x: Numo::DFloat.new(2, 2, 4, 4).seq, w: Numo::DFloat.new(2, 2, 3, 3).seq, options: { stride: 2 } },
      expected: Numo::DFloat[[[[2793.0]], [[7005.0]]], [[[7689.0]], [[22269.0]]]]
    },
    test5: {
      case: { x: Numo::DFloat.new(2, 2, 4, 4).seq, w: Numo::DFloat.new(2, 2, 3, 3).seq, options: { b: Numo::DFloat[10, 33] } },
      expected: Numo::DFloat[[[[2803.0, 2956.0], [3415.0, 3568.0]], [[7038.0, 7515.0], [8946.0, 9423.0]]], [[[7699.0, 7852.0], [8311.0, 8464.0]], [[22302.0, 22779.0], [24210.0, 24687.0]]]]
    },
    test6: {
      case: { x: Numo::DFloat.new(2, 2, 4, 4).seq, w: Numo::DFloat.new(2, 2, 3, 3).seq, options: { b: Numo::DFloat[3, 5], pad: 1 } },
      expected: Numo::DFloat[[[[1199.0, 1799.0, 1919.0, 1267.0], [1884.0, 2796.0, 2949.0, 1926.0], [2316.0, 3408.0, 3561.0, 2310.0], [1427.0, 2075.0, 2159.0, 1383.0]], [[2713.0, 4177.0, 4513.0, 3069.0], [4586.0, 7010.0, 7487.0, 5060.0], [5882.0, 8918.0, 9395.0, 6308.0], [4093.0, 6181.0, 6481.0, 4337.0]]], [[[3887.0, 5639.0, 5759.0, 3699.0], [5340.0, 7692.0, 7845.0, 4998.0], [5772.0, 8304.0, 8457.0, 5382.0], [3347.0, 4763.0, 4847.0, 3047.0]], [[10009.0, 14929.0, 15265.0, 10109.0], [14954.0, 22274.0, 22751.0, 15044.0], [16250.0, 24182.0, 24659.0, 16292.0], [10621.0, 15781.0, 16081.0, 10609.0]]]]
    },
    
  })
  def test_convolution_2d(data)
    test_case = data[:case]
    actual = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(test_case[:x], test_case[:w], **test_case[:options])
    assert_equal(data[:expected], actual.data)
  end
end

