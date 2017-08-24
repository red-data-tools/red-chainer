module Chainer
  module Training
    module Extensions
      class LogReport < Extension
        def initialize(keys: nil, trigger: [1, 'epoch'], postprocess: nil, log_name: 'log')
          @keys = keys
          @trigger = Chainer::Training::Util.get_trigger(trigger)
          @postprocess = postprocess
          @log_name = log_name
          @log = []

          init_summary
        end

        private

        def init_summary
          @summary = DictSummary.new
        end
      end
    end
  end
end
