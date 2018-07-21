# frozen_string_literal: true

require 'chainer/initializers/uniform'

class Chainer::Initializers::UniformTest < Test::Unit::TestCase

  shapes = [[2, 3], [2, 3, 4]]
  dtypes = [ Numo::SFloat, Numo::DFloat ]

  data =  shapes.map.with_index {|shape, i|
            dtypes.map do |dtype|
              ["shape#{i}:#{dtype}", {shape: shape, dtype: dtype}]
            end
          }.flatten(1).to_h

  data(data)
  def test_initializer_cpu(data)
    w = data[:dtype].new(data[:shape])
    initializer = Chainer::Initializers::Uniform.new(scale: 0.1)
    w = initializer.(w)
    assert_equal(w.shape, data[:shape])
    assert_equal(w.class, data[:dtype])
  end

  data(data)
  def test_shaped_initializer(data)
    initializer = Chainer::Initializers::Uniform.new(scale: 0.1, dtype: data[:dtype])
    w = Chainer::Initializers.generate_array(initializer, data[:shape])
    assert_equal(w.shape, data[:shape])
    assert_equal(w.class, data[:dtype])
  end
end
