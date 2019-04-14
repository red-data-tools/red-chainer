require 'benchmark'
require 'chainer'

NUM_ITERATION = 1
XM = Cumo # Numo
Cumo::CUDA::Runtime.cudaDeviceSynchronize

def gen(batch_size: 2, in_channels: 3, out_channels: 2, in_dims: [4, 3], kernel_size: [3, 3])
  kh, kw = kernel_size
  w = XM::SFloat.new(out_channels, in_channels, kh, kw).rand_norm(0, XM::NMath.sqrt(1.0 / (kh * kw * in_channels)))
  b = XM::SFloat.new(out_channels).rand(-1, 1)
  x = XM::SFloat.new(batch_size, in_channels, *in_dims).rand(-1, 1)
  x = Chainer::Variable.new(x)
  w = Chainer::Variable.new(w)
  b = Chainer::Variable.new(b)
  [x, w, b]
end

def conv(x, w, b)
  stride = 2
  pad = 1
  y = Chainer::Functions::Connection::Convolution2DFunction.convolution_2d(
    x, w, b: b, stride: stride, pad: pad, cover_all: false)
  y.data.free
  Cumo::CUDA::Runtime.cudaDeviceSynchronize
end

def bench(r, power)
  x, w, b = gen(batch_size: 32, in_dims: [2**power, 2**power])
  conv(x, w, b) # warm up
  r.report "2**#{power}" do
    NUM_ITERATION.times do
      conv(x, w, b)
    end
  end
end

Benchmark.bm 30 do |r|
  (5..10).each do |power|
    bench(r, power)
  end
end

# Numo
#                                      user     system      total        real
# 2**5                             0.003968   0.003947   0.007915 (  0.007912)
# 2**6                             0.028347   0.002871   0.031218 (  0.031219)
# 2**7                             0.115156   0.018092   0.133248 (  0.133212)
# 2**8                             0.456171   0.062693   0.518864 (  0.518686)
# 2**9                             1.802245   0.247835   2.050080 (  2.049398)
# 2**10                            7.090226   1.040156   8.130382 (  8.127720)
#
# Cumo w/ cuDNN w/ k80
#                                      user     system      total        real
# 2**5                             0.000134   0.000104   0.000238 (  0.000290)
# 2**6                             0.000144   0.000113   0.000257 (  0.000297)
# 2**7                             0.000268   0.000211   0.000479 (  0.000527)
# 2**8                             0.000000   0.001355   0.001355 (  0.001406)
# 2**9                             0.000430   0.004352   0.004782 (  0.004830)
# 2**10                            0.011053   0.007462   0.018515 (  0.018574)
#
# Cumo w/ cuDNN w/ v100
#
#                                      user     system      total        real
# 2**5                             0.000085   0.000079   0.000164 (  0.000178)
# 2**6                             0.000068   0.000062   0.000130 (  0.000149)
# 2**7                             0.000078   0.000071   0.000149 (  0.000168)
# 2**8                             0.000143   0.000129   0.000272 (  0.000291)
# 2**9                             0.000340   0.000280   0.000620 (  0.000637)
# 2**10                            0.001320   0.000816   0.002136 (  0.002153)
