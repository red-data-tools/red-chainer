module Chainer
  module Training
    class Trainer
      def initialize(updater, stop_trigger: nil, out: 'result')
        @updater = updater
        @stop_trigger = Chainer::Training::Util.get_trigger(stop_trigger)
        @observation = {}
        @out = out

        reporter = Reporter.new
        updater.get_all_optimizers().each do |(name, optimizer)|
          reporter.add_observer(name, optimizer.target)
          reporter.add_observers(name, optimizer.target.namedlinks(skipself: true))
        end
        @reporter = reporter

        @done = false
        @extensions = {}

        @start_at = nil
        @snapshot_elapsed_time = 0.0
        @final_elapsed_time = nil

        updater.connect_trainer(self)
      end
    end
  end
end
