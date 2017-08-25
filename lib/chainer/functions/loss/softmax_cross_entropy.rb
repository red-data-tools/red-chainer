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
          log_y = Activation.log_softmax(x)

          if @cache_score
            @y = Numo::NMath.exp(log_y)
          end
          if @class_weight
            shape = x.ndim.times.map { |e| e == 1 ? -1 : 1 } 
            log_y += broadcast_to(@class_weight.reshape(*shape), x.shape)
          end
          log_yd = rollaxis(log_y, 1)
          begin
            log_yd = log_yd.reshape(log_yd.size, -1)
          rescue ArgumentError
          end
          ravel_arr = t.dup.flatten.dup
          ravel_arr = ravel_arr[ravel_arr<0] = 0
          arange_arr = t.class.new(t.size).seq
          log_p = log_yd[ravel_arr, arange_arr]

          t.flatten.dup[(t.dup==@ignore_label)] = 0
          log_p *= t
          if @reduce == 'mean'
            if @normalize
              count = t.ne(@ignore_label).count
            else
              count = x.size
            end
            @coeff = 1.0 / [count, 1].max

            y = log_p.sum(keepdims: true) * (-@coeff)
            [y.reshape(())]
          else
            [-log_p.reshape(t.shape)]
          end
        end

        def backward_cpu(inputs, grad_outputs)
          x, t = inputs
          gloss = grad_outputs[0]

          if self.instance_variable_defined?(:'@y')
            y = @y.dup
          else
            y = Activation.log_softmax(x)
            y = Numo::NMath.exp(y)
          end

          if y.ndim == 2
            gx = y
            t[t<0] = 0
            gx[Numo::DFloat.new(t.size).seq, t] -= 1
            if @class_weight
              shape = x.ndim.times.map { |d| d == 1 ? -1 : 1 }
              c = broadcast_to(@class_weight.reshape(shape), x.shape)
              c = c[Numo::DFloat.new(t.size).seq, t]
              gx *= broadcast_to(t.expand_dims(1), gx.shape)
            end
            gx *= t.flatten.dup[(t.dup==@ignore_label)].reshape(t.size, 1)
          else
            raise 'TODO: ndim > 2 backward'
          end

          if @reduce == 'mean'
            gx *= gloss * @coeff
          else
            raise 'TODO: reduce'
          end
          return [gx, nil]
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
