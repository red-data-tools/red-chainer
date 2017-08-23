module Chainer
  module Functions
    module Loss
      class SoftmaxCrossEntropy < Function
        def self.softmax_cross_entropy(x, t, normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean')
          self.new(normalize, cache_score, class_weight, ignore_label, reduce).(x, t)
        end

        def initialize(normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean')
          @normalize = normalize
          @cache_score = cache_score
          @class_weight = class_weight

          unless class_weight.nil?
            if @class_weight.ndim != 1
              raise ArgumentError 'class_weight.ndim should be 1'
            elsif @class_weight.dtype != Numo::DFloat
              raise ArgumentError 'The dtype of class_weight should be \'Numo::DFloat\''
            elsif @class_weight.kind_of?(Chainer::Variable)
              raise ArgumentError 'class_weight should be a Numo::NArray, not a chainer.Variable'
            end
          end

          @ignore_label = ignore_label
          unless ['mean', 'no'].include?(reduce)
            raise ArgumentError "only 'mean' and 'no' are valid for 'reduce', but #{reduce} is given"
          end

          @reduce = reduce
        end

        def forward_cpu(inputs)
          x, t = inputs
          log_y = Activation._log_softmax(x)

          if @cache_score
            @y = Numo::NMath.exp(log_y)
          end
          if @class_weight
            shape = x.ndim.times.map { |e| e == 1 ? -1 : 1 } 
            log_y += broadcast_to(@class_weight.reshape(*shape), x.shape)
          end
          log_yd = rollaxis(log_y, 1)
          log_yd = log_yd.reshape(len(log_yd), -1)

          ravel_arr = t.flatten.dup
          ravel_arr = ravel_arr[ravel_arr<0] = 0
          arange_arr = t.class.new(t.size).seq
          log_p = log_yd[ravel_arr, arange_arr]

          log_p *= (t.flatten.dup != @ignore_label)
          if @reduce == 'mean'
            if @normalize
              count = t.ne(@ignore_label).count
            else
              count = x.size
            end
            @coeff = 1.0 / [count, 1].max

            y = log_p.sum(keepdims: true) * (-@coeff)
            [y.reshape([])]
          else
            [-log_p.reshape(t.shape)]
          end
        end

        private

        def broadcast_to(array, shape)
          array.class.tile(array, shape[0]).reshape(*shape)
        end

        def rollaxis(y, axis, start: 0)
          axes = (0...y.ndim).to_a
          axes.delete_at(axis)
          axes.insert(start, axis)
          y.transpose(*axes)
        end
      end
    end
  end
end
