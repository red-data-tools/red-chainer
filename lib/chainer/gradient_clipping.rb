# Optimizer hook function for gradient clipping.
#
# This hook function scales all gradient arrays to fit to the defined L2 norm
# threshold.
#
# Args:
#     threshold (float): L2 norm threshold.
#
# Attributes:
#     threshold (float): L2 norm threshold of gradient norm.
module Chainer
  class GradientClipping
    def self.name
      'GradientClipping'
    end

    def initialize(threshold)
      @threshold = threshold
    end

    def call(opt)
      params = []
      opt.target.params(include_uninit: false) { |param| params << param }
      norm = Numo::NMath.sqrt(params.map{|param| param.grad.reshape(param.grad.shape.inject(&:*))}.inject(0.0) {|sum, x| sum + x.dot(x) })
      rate = @threshold / norm[0]
      if rate < 1
        params.each do |param|
          param.grad *= rate
        end
      end
    end
  end
end
