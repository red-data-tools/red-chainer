# frozen_string_literal: true

class Chainer::Links::Connection::EmbedIDTest < Test::Unit::TestCase
  data = {
    test1: {x_data: [0, 1, 0], ignore_label: nil},
    test2: {x_data: [[0, 1, 0], [1, 0, 1]], ignore_label: nil},
    test3: {x_data: [0, 1, -1], ignore_label: -1},
    test4: {x_data: [[0, 1, -1], [-1, 0, 1]], ignore_label: -1},
  }

  def setup
    @link = Chainer::Links::Connection::EmbedID.new(3, 2, ignore_label: data[:ignore_label])
    @link.cleargrads

    @w = @link.w.data.copy()
    @x = xm::Int32[*data[:x_data]]
    y_shape = @x.shape + [2]
    @gy = xm::SFloat.new(*y_shape).rand(-1, 1)
  end

  data(data)
  def test_forward(data)
    x = Chainer::Variable.new(xm::Int32.cast(data[:x_data]))
    y = @link.(x)
    assert_equal(y.data.class, xm::SFloat)
    y_expect = @gy.dup

    @x.shape.reduce(&:*).times do |i|
      ndindex = @x.shape.size.times.reduce([]) do |ndi, j|
        ndi << (i / @x.shape.drop(j+1).reduce(1, &:*)) % @x.shape[j]
      end
      y_expect[*ndindex, true] = @x[*ndindex] == -1 ? 0 : @w[@x[*ndindex], true]
    end

    assert_equal(y_expect.to_a, y.data.to_a)
  end

  data(data)
  def test_backward(data)
    Chainer::check_backward(@link, @x, @gy, @link.w, atol: 1e-4)
  end
end

