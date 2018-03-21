# frozen_string_literal: true

class Chainer::Links::Connection::Convolution2DTest < Test::Unit::TestCase
  data({
    test1: {
      case: { x: Numo::DFloat.new(1, 3, 10, 10).seq, in_channels: 3, out_channels: 7, ksize: 5, options: {} },
      expected: [1, 7, 6, 6]
    },
    test1: {
      case: { x: Numo::DFloat.new(3, 3, 6, 6).seq, in_channels: nil, out_channels: 4, ksize: 3, options: { stride: 3, pad: 3, initial_w: Numo::DFloat.new(4, 3, 3, 3).seq } },
      expected: [3, 4, 4, 4]
    },
  })
  def test_get_conv_outsize(data)
    test_case = data[:case]
    l = Chainer::Links::Connection::Convolution2D.new(test_case[:in_channels], test_case[:out_channels], test_case[:ksize], **test_case[:options])
    assert_equal(data[:expected], l.(test_case[:x]).shape)
  end
end

