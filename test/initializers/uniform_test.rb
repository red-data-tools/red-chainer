# frozen_string_literal: true

require 'chainer/initializers/uniform'

class Chainer::Initializers::UniformTest < Test::Unit::TestCase

  data(:shape, [[2, 3], [2, 3, 4]],        keep: true)
  data(:dtype, [ xm::SFloat, xm::DFloat ], keep: true)

  def test_initializer()
    w = data[:dtype].new(data[:shape])
    initializer = Chainer::Initializers::Uniform.new(scale: 0.1)
    w = initializer.(w)
    assert_equal(w.shape, data[:shape])
    assert_equal(w.class, data[:dtype])
  end

  def test_shaped_initializer()
    initializer = Chainer::Initializers::Uniform.new(scale: 0.1, dtype: data[:dtype])
    w = Chainer::Initializers.generate_array(initializer, data[:shape])
    assert_equal(w.shape, data[:shape])
    assert_equal(w.class, data[:dtype])
  end
end
