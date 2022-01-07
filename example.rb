require 'chainer'

x = Numo::DFloat.new(2,3,4).seq
pp Chainer::Utils::Array.take(x, [1, 2], axis: 1)
