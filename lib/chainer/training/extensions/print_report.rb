module Chainer
  module Training
    module Extensions
      class PrintReport < Extension
        def initialize(entries, log_report: 'LogReport', out: STDOUT)
          @entries = entries
          @log_report = log_report
          @out = out

          @log_len = 0

          # format information
          entry_widths = entries.map { |s| [10, s.size].max }

          @header = entries.map { |e| "#{e}" }.join('  ')

          templates = []
          @templates = templates
        end
      end
    end
  end
end
