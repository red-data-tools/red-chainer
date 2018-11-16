module Chainer
  module Functions
    module Evaluation
      class Accuracy < Function
        def self.accuracy(y, t, ignore_label: nil)
          self.new(ignore_label: ignore_label).(y, t)
        end

        def initialize(ignore_label: nil)
          @ignore_label = ignore_label
        end

        def forward(inputs)
          y, t = inputs
          xm = Chainer.get_array_module(*inputs)
          if @ignore_label
            mask = t.eq(@ignore_label)
            ignore_cnt = mask.count

            pred = y.max_index(axis: 1) - xm::Int32.new(y.shape[0]).seq(0, y.shape[1])
            pred = pred.reshape(*t.shape)
            pred[mask] = @ignore_label
            count = pred.eq(t).count - ignore_cnt

            total = t.size - ignore_cnt

            if total == 0
              [y.class.cast(0.0)]
            else
              [y.class.cast(count.to_f / total)]
            end
          else
            pred = y.max_index(axis: 1) - xm::Int32.new(y.shape[0]).seq(0, y.shape[1])
            pred = pred.reshape(*t.shape)

            [y.class.cast(y.class[pred.eq(t)].mean)]
          end
        end
      end
    end
  end
end
