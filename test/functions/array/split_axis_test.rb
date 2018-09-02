# frozen_string_literal: true

class Chainer::Functions::Array::SplitAxisTest < Test::Unit::TestCase
  data = Chainer::Testing::Parameterize.product_dict(
    [
      { 'shape': [2, 7, 3], 'axis': 1, 'ys_section': [2, 5],
        'slices': [[true, 0...2, true], [true, 2...5, true], [true, 5...7, true]]},
      { 'shape': [7, 3], 'axis': 0, 'ys_section': [2, 5],
        'slices': [[0...2, true], [2...5, true], [5...7, true]]},
      { 'shape': [2, 9, 3], 'axis': 1, 'ys_section': 3,
        'slices': [[true, 0...3, true], [true, 3...6, true], [true, 6...9, true]]},
      { 'shape': [2, 6, 3], 'axis': 1, 'ys_section': 3,
        'slices': [[true, 0...2, true], [true, 2...4, true], [true, 4...6, true]]},
      { 'shape': [2], 'axis': 0, 'ys_section': [1],
        'slices': [0...1, 1...2]},
      { 'shape': [2], 'axis': 0, 'ys_section': [],
        'slices': [true]},
      # { 'shape': [2, 7, 3], 'axis': 1, 'ys_section': [0],
      #   'slices': [[true, [], true], [true, 0...7, true]]},
    ], [
      { 'dtype': Numo::SFloat },
      { 'dtype': Numo::DFloat },
    ]
  )


  def _setup(data)
    @x = data[:dtype].new(data[:shape]).seq
    @ys = data[:slices].map{|s| @x[*s] }
  end

  data(data)
  def test_forward(data)
    _setup(data)
    x = Chainer::Variable.new(@x)
    ys = Chainer::Functions::Array::SplitAxis.split_axis(
            x, data[:ys_section], data[:axis], force_tuple: true)
    @ys.zip(ys).each do |yd, y|
      assert_equal(y.data.class, data[:dtype])
      Chainer::Testing.assert_allclose(yd, y.data, atol: 0, rtol: 0)
    end
  end

  data(data)
  def check_backward(data)
    x = Chainer::Variable.new(@x)
    ys = Chainer::Functions::Array::SplitAxis.split_axis(
            x, data[:ys_section], data[:axis], force_tuple: true)
    ys.each { |y| y.grad = y.data }

    ys[0].backward

    Chainer::Testing.assert_allclose(x.data, x.grad, atol: 0, rtol: 0)
  end
end
