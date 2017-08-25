module Chainer
  module Training
    module Util
      def self.get_trigger(trigger)
        if trigger.nil?
          false
        else
          Triggers::IntervalTrigger.new(*trigger)
        end
      end
    end
  end
end
