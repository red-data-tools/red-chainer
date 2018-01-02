module Chainer
  module Training
    module Triggers
      class IntervalTrigger
        attr_reader :period, :unit, :count

        def initialize(period, unit)
          @period = period
          @unit = unit
          @count = 0
        end

        def call(trainer)
          updater = trainer.updater
          if @unit == 'epoch'
            prev = @count
            @count = updater.epoch_detail.div(@period).floor
            prev != @count
          else
            iteration = updater.iteration
            iteration > 0 && iteration % @period == 0
          end
        end
      end
    end
  end
end
