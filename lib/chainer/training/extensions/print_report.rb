module Chainer
  module Training
    module Extensions
      class PrintReport < Extension
        def initialize(entries, log_report: 'Chainer::Training::Extensions::LogReport', out: STDOUT)
          @entries = entries
          @log_report = log_report
          @out = out

          @log_len = 0 # number of observations already printed

          # format information
          entry_widths = entries.map { |s| [10, s.size].max }

          templates = []
          header = []
          entries.zip(entry_widths).each do |entry, w|
            header << sprintf("%-#{w}s", entry)
            templates << [entry, "%-#{w}g  ", ' ' * (w + 2)]
          end
          @header = header.join('  ') + "\n"
          @templates = templates
        end

        def call(trainer)
          if @header
            @out.write(@header)
            @header = nil
          end
         
          if @log_report.is_a?(String)
            log_report = trainer.get_extension(@log_report)
          elsif @log_report.is_a?(LogReport)
            log_report.(trainer)
          else
            raise TypeError, "log report has a wrong type #{log_report.class}"
          end

          log = log_report.log
          while log.size > @log_len
            @out.write("\033[J")
            print(log[@log_len])
            @log_len += 1
          end
        end

        def serialize(serializer)
          if @log_report.is_a?(Chainer::Training::Extensions::LogReport)
            @log_report.serialize(serializer['_log_report'])
          end
        end

        private

        def print(observation)
          @templates.each do |entry, template, empty|
            if observation.keys.include?(entry)
              @out.write(sprintf(template, observation[entry]))
            else
              @out.write(empty)
            end
          end
          @out.write("\n")
        end
      end
    end
  end
end
