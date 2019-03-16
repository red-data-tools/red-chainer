class Chainer::Functions::Pooling::MaxPooling2DTest < Test::Unit::TestCase
  data(:dtype, [xm::SFloat, xm::DFloat], keep: true)
  data(:cover_all, [true, false], keep: true)

  def setup
    @dtype = data[:dtype]
    @cover_all = data[:cover_all]

    x = @dtype.new(2, 3, 4, 3).seq.to_a.shuffle
    @x = @dtype[*x]
    @x = 2 * @x / @x.size - 1

    if @cover_all
      @gy = @dtype.new(2, 3, 3, 2).rand(-1, 1)
    else
      @gy = @dtype.new(2, 3, 2, 2).rand(-1, 1)
    end

    @ggx = @dtype.new(2, 3, 4, 3).rand(-1, 1)
  end

  def check_forward(x_data, use_cudnn: 'always')
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(x, 3, stride: 2, pad: 1, cover_all: @cover_all)
    assert_equal(y.data.class, @dtype)

    y_data = y.data
    assert_equal(@gy.shape, y_data.shape)

    2.times.each do |k|
      3.times.each do |c|
        x = @x[k, c, false]
        if @cover_all
          expect = xm::DFloat[
            [x[0...2, 0...2, false].max, x[0...2, 1...3, false].max],
            [x[1...4, 0...2, false].max, x[1...4, 1...3, false].max],
            [x[1...4, 0...2, false].max, x[3...4, 1...3, false].max],
          ]
        else
          expect = xm::DFloat[
            [x[0...2, 0...2, false].max, x[0...2, 1...3, false].max],
            [x[1...4, 0...2, false].max, x[1...4, 1...3, false].max],
          ]
        end
        Chainer::Testing.assert_allclose(expect, y_data[k, c, false])
      end
    end
  end

  def test_forward
    check_forward(@x)
  end

  def check_backward(x_data, y_grad, use_cudnn: 'always')
    func = -> (x) do
      Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(x, 3, stride: 2, pad: 1, cover_all: @cover_all)
    end
    Chainer::check_backward(func, x_data, y_grad, dtype: xm::DFloat, atol: 1e-4, rtol: 1e-3)
  end

  def test_backward
    check_backward(@x.dup, @gy.dup)
  end

  def check_double_backward(x_data, y_grad, x_grad_grad, use_cudnn: 'always')
    func = -> (x) do
      y = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(x, 3, stride: 2, pad: 1, cover_all: @cover_all)
      y * y
    end
    Chainer::check_double_backward(func, x_data, y_grad, x_grad_grad, dtype: xm::DFloat, atol: 1e-4, rtol: 1e-3)
  end

  def test_double_backward
    check_double_backward(@x, @gy, @ggx)
  end
end
