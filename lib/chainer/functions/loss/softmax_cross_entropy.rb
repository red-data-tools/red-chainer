module Chainer
  module Functions
    module Loss
      class SoftmaxCrossEntropy < Function
        def self.softmax_cross_entropy(x, t, normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean')
          self.new(normalize: normalize, cache_score: cache_score, class_weight: class_weight, ignore_label: ignore_label, reduce: reduce).(x, t)
        end

        def initialize(normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean')
          @normalize = normalize
          @cache_score = cache_score
          @class_weight = class_weight

          unless class_weight.nil?
            xm = Chainer.get_array_module(@class_weight)
            if @class_weight.ndim != 1
              raise ArgumentError, 'class_weight.ndim should be 1'
            elsif (@class_weight.class != xm::DFloat) and (@class_weight.class != xm::SFloat)
              raise ArgumentError, "The dtype of class_weight should be 'DFloat' or 'SFloat'"
            elsif @class_weight.kind_of?(Chainer::Variable)
              raise ArgumentError, 'class_weight should be a NArray, not a chainer.Variable'
            end
          end

          @ignore_label = ignore_label
          unless ['mean', 'no'].include?(reduce)
            raise ArgumentError, "only 'mean' and 'no' are valid for 'reduce', but #{reduce} is given"
          end

          @reduce = reduce
        end

        def forward(inputs)
          x, t = inputs
          log_y = Activation._log_softmax(x)

          if @cache_score
            xm = Chainer.get_array_module(log_y)
            @y = xm::NMath.exp(log_y)
          end
          if @class_weight
            shape = x.ndim.times.map { |e| e == 1 ? true : 1 }
            log_y *= Chainer::Utils::Array.broadcast_to(@class_weight.reshape(*shape), x.shape)
          end
          log_yd = Chainer::Utils::Array.rollaxis(log_y, 1)
          begin
            log_yd = log_yd.reshape(log_yd.shape[0], true)
          rescue ArgumentError
          end
          ravel_arr = t.dup.flatten.dup
          ravel_arr[ravel_arr<0] = 0
          arange_arr = t.class.new(t.size).seq

          # https://github.com/chainer/chainer/blob/v2.0.2/chainer/functions/loss/softmax_cross_entropy.py#L79
          log_p = []
          ravel_arr.each_with_index do |r, i|
            log_p << log_yd[r, i]
          end
          log_p = log_yd.class.[](*log_p)
          log_p[t.flatten.dup.eq(@ignore_label)] = 0

          if @reduce == 'mean'
            if @normalize
              count = t.ne(@ignore_label).count
            else
              count = x.shape[0]
            end
            @coeff = 1.0 / [count, 1].max
            y = log_p.sum(keepdims: true) * (-@coeff)
            [y.class.cast(y[0])]
          else
            [-log_p.reshape(*t.shape)]
          end
        end

        def backward(inputs, grad_outputs)
          x, t = inputs
          gloss = grad_outputs[0]

          if self.instance_variable_defined?(:'@y')
            y = @y.dup
          else
            y = Activation._log_softmax(x)
            xm = Chainer.get_array_module(y)
            y = xm::NMath.exp(y)
          end

          if y.ndim == 2
            gx = y
            t.class.new(t.shape[0]).seq(0).to_a.zip(t.class.maximum(t, 0).to_a).each{|v| gx[*v] -= 1}

            if @class_weight
              shape = x.ndim.times.map { |d| d == 1 ? true : 1 }
              c = Chainer::Utils::Array.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c.class.cast(t.class.new(t.shape[0]).seq.to_a.zip(t.class.maximum(t, 0).to_a).map{|v| c[*v]})
              gx *= Chainer::Utils::Array.broadcast_to(c.expand_dims(1), gx.shape)
            end

            bit = t.flatten.dup
            bit[t.ne(@ignore_label)] = 1
            bit[bit.ne(1)] = 0
            gx *= bit.reshape(t.shape[0], 1)
          else
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.

            n_unit = t.size / t.shape[0]
            gx = y.reshape(y.shape[0], y.shape[1], true)
            fst_index = Numo::Int32.new(t.size).seq(0) / n_unit
            trd_index = Numo::Int32.new(t.size).seq(0) % n_unit
            fst_index.to_a.zip(t.class.maximum(t.flatten.dup, 0).to_a, trd_index.to_a).each{|v| gx[*v] -= 1}
            if @class_weight
              shape = x.ndim.times.map{|d| d == 1 ? true : 1}
              c = Chainer::Utils::Array.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c.reshape(*gx.shape)
              c = c.class.cast(fst_index.to_a.zip(t.class.maximum(t.flatten.dup, 0).to_a, trd_index.to_a).map{|v| c[*v]})
              c = c.reshape(y.shape[0], 1, true)
              gx *= Chainer::Utils::Array.broadcast_to(c, gx.shape)
            end
            gx *= (t.ne @ignore_label).reshape(t.shape[0], 1, true)
            gx = gx.reshape(*y.shape)
          end

          if @reduce == 'mean'
            gx *= gloss * @coeff
          else
            gx *= gloss[true,:- , false]
          end
          return [gx, nil]
        end
      end
    end
  end
end
