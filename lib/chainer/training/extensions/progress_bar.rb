module Chainer
  module Training
    module Extensions
      class ProgressBar < Extension
        def initialize(training_length: nil, update_interval: 100,  bar_length: 50, out: STDOUT)
          @training_length = training_length
          @status_template = nil
          @update_interval = update_interval
          @bar_length = bar_length
          @out = out
          @recent_timing = []
        end
      end
    end
  end
end
