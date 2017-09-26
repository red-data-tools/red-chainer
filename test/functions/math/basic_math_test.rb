require 'chainer'
require 'numo/narray'

class Chainer::Functions::Math::BasicMathTest < Test::Unit::TestCase
  test("Neg#forward") do
    x = Chainer::Variable.new(Numo::DFloat[[-1, 0],[1, 2]])
    assert_equal(Numo::DFloat[[1,0],[-1,-2]], (-x).data)
  end
end
