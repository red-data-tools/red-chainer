module Chainer
  module Functions
    module Loss
      class SoftmaxCrossEntropy < Function
        def self.softmax_cross_entropy(x, t, normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean', enable_double_backprop: false)
          if enable_double_backprop
            self.double_backward_softmax_cross_entropy(x, t, normalize, class_weight, ignore_label, reduce)
          else
            self.new(normalize: normalize, cache_score: cache_score, class_weight: class_weight, ignore_label: ignore_label, reduce: reduce).(x, t)
          end
        end

        def self.double_backward_softmax_cross_entropy(x, t, normalize, class_weight, ignore_label, reduce)
          if t.is_a?(Chainer::Variable)
            t =  t.data
          end

          self.check_class_weight_option(class_weight)
          self.check_reduce_option(reduce)

          loss = -Activation::LogSoftmax.log_softmax(x)

          if class_weight
            shape = x.ndim.times.map { |d| d != 1 ? 1 : class_weight.shape[-1] }
            class_weight = Chainer::Functions::Array::BroadcastTo.broadcast_to(class_weight.reshape(*shape), x.shape)
            loss = loss * class_weight
          end

          dtype = x.is_a?(Chainer::Variable) ? x.dtype : x.class
          in_use = t.ne(ignore_label).cast_to(dtype)

          loss = Chainer::Functions::Array::Rollaxis.rollaxis(loss, 1, start:  loss.ndim)

          # TODO: loss = chainer.functions.reshape(loss, (-1, loss.shape[-1]))
          shape = loss.shape
          last_shape = shape.pop
          loss = Chainer::Functions::Array::Reshape.reshape(loss, [shape.inject(:*), last_shape])

          # Replace ignore_label value with one valid for F.select_item below.
          t = t.clip(0, loss.shape[1] - 1)

          loss = Chainer::Functions::Array::SelectItem.select_item(loss, t.flatten.dup)
          loss = Chainer::Functions::Array::Reshape.reshape(loss, t.shape)

          loss = loss * in_use

          if reduce == "mean"
            count = normalize ? in_use.sum : x.shape.first
            count = [count, 1.0].max
            loss = loss * (1.0 / count)
            return Chainer::Functions::Math::Sum.sum(loss)
          else
            return loss
          end
        end

        def initialize(normalize: true, cache_score: true, class_weight: nil, ignore_label: -1, reduce: 'mean')
          @normalize = normalize
          @cache_score = cache_score
          self.class.check_class_weight_option(class_weight)
          @class_weight = class_weight

          @ignore_label = ignore_label

          self.class.check_reduce_option(reduce)
          @reduce = reduce
        end

        def forward(inputs)
          xm = Chainer.get_array_module(*inputs)
          x, t = inputs
          log_y = Activation._log_softmax(x)

          if @cache_score
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

          log_p = log_yd[t.class.maximum(t.flatten, 0), t.class.new(t.size).seq].diagonal
          if @ignore_label
            t_valid= t.ne(@ignore_label)
            log_p *= t_valid.flatten
          end

          if @reduce == 'mean'
            if @normalize and t_valid
              @coeff = 1.0 / log_p.class.maximum(Chainer::Utils::Array.force_array(t_valid.count), 1)
            else
              count = x.shape[0]
              @coeff = 1.0 / [count, 1].max
            end
            y = log_p.sum(keepdims: true) * (-@coeff)
            [y.class.cast(y[0])]
          else
            [-log_p.reshape(*t.shape)]
          end
        end

        def backward(inputs, grad_outputs)
          xm = Chainer.get_array_module(*(inputs + grad_outputs))
          x, t = inputs
          gloss = grad_outputs[0]

          if self.instance_variable_defined?(:'@y')
            y = @y.dup
          else
            y = Activation._log_softmax(x)
            y = xm::NMath.exp(y)
          end

          if y.ndim == 2
            gx = y
            # TODO(sonots): Avoid to_a especially in Cumo to improve performance
            t.class.new(t.shape[0]).seq(0).to_a.zip(t.class.maximum(t, 0).to_a).each{|v| gx[*v] -= 1}

            if @class_weight
              shape = x.ndim.times.map { |d| d == 1 ? true : 1 }
              c = Chainer::Utils::Array.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c[t.class.new(t.shape[0]).seq, t.class.maximum(t, 0)].diagonal.dup
              gx *= Chainer::Utils::Array.broadcast_to(c.expand_dims(1), gx.shape)
            end

            if @ignore_label
              gx *= (t.ne @ignore_label).reshape(t.shape[0], 1)
            end
          else
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.

            n_unit = t.size / t.shape[0]
            gx = y.reshape(y.shape[0], y.shape[1], true)
            fst_index = xm::Int32.new(t.size).seq(0) / n_unit
            trd_index = xm::Int32.new(t.size).seq(0) % n_unit
            # TODO(sonots): Avoid to_a especially in Cumo to improve performance
            fst_index.to_a.zip(t.class.maximum(t.flatten.dup, 0).to_a, trd_index.to_a).each{|v| gx[*v] -= 1}
            if @class_weight
              shape = x.ndim.times.map{|d| d == 1 ? true : 1}
              c = Chainer::Utils::Array.broadcast_to(@class_weight.reshape(*shape), x.shape)
              c = c.reshape(*gx.shape)
              c = c[fst_index, t.class.maximum(t.flatten.dup, 0), trd_index].diagonal.diagonal.dup
              c = c.reshape(y.shape[0], 1, true)
              gx *= Chainer::Utils::Array.broadcast_to(c, gx.shape)
            end
            if @ignore_label
              gx *= (t.ne @ignore_label).reshape(t.shape[0], 1, true)
            end
            gx = gx.reshape(*y.shape)
          end

          if @reduce == 'mean'
            gx *= gloss * @coeff
          else
            gx *= gloss[true,:- , false]
          end
          return [gx, nil]
        end

        def self.check_class_weight_option(class_weight)
          return if class_weight.nil?

          xm = Chainer.get_array_module(class_weight)
          if class_weight.ndim != 1
            raise ArgumentError, 'class_weight.ndim should be 1'
          elsif (class_weight.class != xm::DFloat) and (class_weight.class != xm::SFloat)
            raise ArgumentError, "The dtype of class_weight should be 'DFloat' or 'SFloat', but was #{class_weight.class}"
          elsif class_weight.kind_of?(Chainer::Variable)
            raise ArgumentError, 'class_weight should be a NArray, not a chainer.Variable'
          end
        end

        def self.check_reduce_option(reduce)
          unless ['mean', 'no'].include?(reduce)
            raise ArgumentError, "only 'mean' and 'no' are valid for 'reduce', but #{reduce} is given"
          end
        end
      end
    end
  end
end
