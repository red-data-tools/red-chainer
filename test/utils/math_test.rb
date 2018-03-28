# frozen_string_literal: true

class Chainer::Utils::MathTest < Test::Unit::TestCase
  data({
    test1: {
      case: {
        a: Numo::DFloat.new(2, 5, 2).seq,
        b: Numo::DFloat.new(5, 2).seq,
        axes: 2
      },
      expected: Numo::DFloat[285.0, 735.0]
    },
    test2: {
      case: {
        a: Numo::DFloat.new(1, 3, 4).seq,
        b: Numo::DFloat.new(5, 3).seq,
        axes: [1, 1]
      },
      expected: Numo::DFloat[[[20.0, 56.0, 92.0, 128.0, 164.0], [23.0, 68.0, 113.0, 158.0, 203.0], [26.0, 80.0, 134.0, 188.0, 242.0], [29.0, 92.0, 155.0, 218.0, 281.0]]]
    },
    test3: {
      case: {
        a: Numo::DFloat.new(1, 3, 4, 2).seq,
        b: Numo::DFloat.new(1, 3, 1, 4).seq,
        axes: [[1, 2], [1, 3]]
      },
      expected: Numo::DFloat[[[[1012.0]], [[1078.0]]]]
    }
  })
  def test_tensordot(data)
    actual = Chainer::Utils::Math.tensordot(data[:case][:a], data[:case][:b], data[:case][:axes])
    assert_equal(data[:expected], actual)

  end
end
