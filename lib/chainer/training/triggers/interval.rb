module Chainer
  module Training
    module Triggers
      class IntervalTrigger
        attr_reader :period, :unit, :count

        def initialize(period, unit)
          @period = period
          @unit = unit
          @count = 0

          @previous_iteration = 0
          @previous_epoch_detail = 0.0
        end

        def call(trainer)
          updater = trainer.updater
          if @unit == 'epoch'
            epoch_detail = updater.epoch_detail
            previous_epoch_detail = @previous_epoch_detail

            if previous_epoch_detail < 0
              previous_epoch_detail = updater.previous_epoch_detail
            end

            @count = epoch_detail.div(@period).floor

            fire = previous_epoch_detail.div(@period).floor != epoch_detail.div(@period).floor
          else
            iteration = updater.iteration
            previous_iteration = @previous_iteration
            if previous_iteration < 0
                previous_iteration = iteration - 1
            end
            fire = previous_iteration.div(@period).floor != iteration.div(@period).floor
          end

          # save current values
          @previous_iteration = updater.iteration
          @previous_epoch_detail = updater.epoch_detail
        
          fire
        end

        def serialize(serializer)
          @previous_iteration = serializer.('previous_iteration', @previous_iteration)
          @previous_epoch_detail = serializer.('previous_epoch_detail', @previous_epoch_detail)
        end
      end
    end
  end
end
