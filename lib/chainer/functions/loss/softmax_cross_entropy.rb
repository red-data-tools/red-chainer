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
              raise ArgumentError, 'class_weight.ndim should be 1'
            elsif (@class_weight.class != Numo::DFloat) and (@class_weight.class != Numo::SFloat)
              raise ArgumentError, "The dtype of class_weight should be 'Numo::DFloat' or 'Numo::SFloat'"
            elsif @class_weight.kind_of?(Chainer::Variable)
              raise ArgumentError, 'class_weight should be a Numo::NArray, not a chainer.Variable'
            end
          end

          @ignore_label = ignore_label
          unless ['mean', 'no'].include?(reduce)
            raise ArgumentError, "only 'mean' and 'no' are valid for 'reduce', but #{reduce} is given"
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
            shape = x.ndim.times.map { |e| e == 1 ? true : 1 }
            log_y *= Chainer::Functions::Loss.broadcast_to(@class_weight.reshape(*shape), x.shape)
          end
          log_yd = Chainer::Functions::Loss.rollaxis(log_y, 1)
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

        def backward_cpu(inputs, grad_outputs)
          x, t = inputs
          gloss = grad_outputs[0]

          if self.instance_variable_defined?(:'@y')
            y = @y.dup
          else
            y = Activation._log_softmax(x)
            y = Numo::NMath.exp(y)
          end

          if y.ndim == 2
            gx = y
            Numo::DFloat.new(t.shape[0]).seq(0).to_a.zip(Numo::DFloat.maximum(t, 0).to_a).each{|v| gx[*v] -= 1}

            if @class_weight
              shape = x.ndim.times.map { |d| d == 1 ? true : 1 }
              c = Chainer::Functions::Loss.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c.class.cast(Numo::DFloat.new(t.shape[0]).seq.to_a.zip(Numo::DFloat.maximum(t, 0).to_a).map{|v| c[*v]})
              gx *= Chainer::Functions::Loss.broadcast_to(c.expand_dims(1), gx.shape)
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
            fst_index.to_a.zip(Numo::DFloat.maximum(t.flatten.dup, 0).to_a, trd_index.to_a).each{|v| gx[*v] -= 1}
            if @class_weight
              shape = x.ndim.times.map{|d| d == 1 ? true : 1}
              c = Chainer::Functions::Loss.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c.reshape(*gx.shape)
              c = c.class.cast(fst_index.to_a.zip(Numo::DFloat.maximum(t.flatten.dup, 0).to_a, trd_index.to_a).map{|v| c[*v]})
              c = c.reshape(y.shape[0], 1, true)
              gx *= Chainer::Functions::Loss.broadcast_to(c, gx.shape)
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

      def rollaxis(y, axis, start: 0)
        axes = (0...y.ndim).to_a
        axes.delete_at(axis)
        axes.insert(start <= axes.size ? start : -1, axis)
        y.transpose(*axes)
      end

      def broadcast_to(array, shape)
        if array.shape.size > shape.size
           raise TypeError, "Shape of data  mismatch\n array.shape.size(#{array.shape.size}) > shape.size(#{shape.size})"
        end

        tile_shape = []
        shape_check = shape[-array.shape.size..-1]
        shape_check.each_with_index{|s, i|
          if array.shape[i] == 1
            tile_shape << s
          elsif array.shape[i] == s
            tile_shape << 1
          else
            raise TypeError, "Shape of data  mismatch\n#{array.shape} != #{shape}"
          end
        }

        array.tile(*shape[0...-array.shape.size], *tile_shape)
      end

      module_function :rollaxis, :broadcast_to
    end
  end
end
