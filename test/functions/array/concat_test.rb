# frozen_string_literal: true

module Chainer::Functions::Array
  class ConcatTest < Test::Unit::TestCase
    data = Chainer::Testing::Parameterize.product_dict(
      [
        {shape: [2, 7, 3], axis: 1,
          slices: [[true, 0...2, true], [true, 2...5, true], [true, 5...7, true]]},
        {shape: [7, 3], axis: 0,
          slices: [[0...2, true], [2...5, true], [5...7, true]]},
        {shape: [2], axis: 0, slices: [[0...1], [1...2]]},
        #{shape: [2], axis: 0, slices: [[]]},
        {shape: [2, 7, 3], axis: 1,
          slices: [[true, 0...2, true], [true, 2...5, true], [true, 5...7, true]]},
        {shape: [2, 7, 3], axis: -2,
          slices: [[true, 0...2, true], [true, 2...5, true], [true, 5...7, true]]},
      ], [
        {dtype: Numo::SFloat},
        {dtype: Numo::DFloat},
      ]
    )

    def _setup(data)
      @y = data[:dtype].new(data[:shape]).seq
      @xs = data[:slices].map{|s| @y[*s] }
    end

    data(data)
    def test_forward(data)
      _setup(data)
      xs = @xs.map {|x_data| Chainer::Variable.new(x_data) }
      y = Chainer::Functions::Array::Concat.concat(xs, axis: data[:axis])
      assert_equal(y.data.class, data[:dtype])
      Chainer::Testing.assert_allclose(@y, y.data, atol: 0, rtol: 0)
    end
  end
end
