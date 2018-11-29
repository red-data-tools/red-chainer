require 'erb'
require 'time'

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
          @out.sync = true
          @recent_timing = []
        end

        def call(trainer)
          if @training_length.nil?
            t = trainer.stop_trigger
            raise TypeError, "cannot retrieve the training length #{t.class}" unless t.is_a?(Chainer::Training::Triggers::IntervalTrigger)
            @training_length = [t.period, t.unit]
          end

          if @status_template.nil?
            @status_template = ERB.new("<%= sprintf('%10d', self.iteration) %> iter, <%= self.epoch %> epoch / #{@training_length[0]} #{@training_length[1]}s\n")
          end

          length, unit = @training_length
          iteration = trainer.updater.iteration

          # print the progress bar according to interval
          return unless iteration % @update_interval == 0

          epoch = trainer.updater.epoch_detail
          now = Time.now.to_f

          @recent_timing << [iteration, epoch, now]
          @out.write("\033[J")

          if unit == 'iteration'
            rate = iteration.to_f / length
          else
            rate = epoch.to_f / length
          end

          marks = '#' * (rate * @bar_length).to_i
          @out.write(sprintf("     total [%s%s] %6.2f%\n", marks, '.' * (@bar_length - marks.size), rate * 100))

          epoch_rate = epoch - epoch.to_i
          marks = '#' * (epoch_rate * @bar_length).to_i
          @out.write(sprintf("this epoch [%s%s] %6.2f%\n", marks, '.' * (@bar_length - marks.size), epoch_rate * 100))

          status = @status_template.result(trainer.updater.bind)
          @out.write(status)

          old_t, old_e, old_sec = @recent_timing[0]
          span = now - old_sec

          if span.zero?
            speed_t = Float::INFINITY
            speed_e = Float::INFINITY
          else
            speed_t = (iteration - old_t) / span
            speed_e = (epoch - old_e) / span
          end

          if unit == 'iteration'
            estimated_time = (length - iteration) / speed_t
          else
            estimated_time = (length - epoch) / speed_e
          end

          @out.write(sprintf("%10.5g iters/sec. Estimated time to finish: %s.\n", speed_t, (Time.parse("1991/01/01") + (estimated_time)).strftime("%H:%m:%S")))

          # move the cursor to the head of the progress bar
          @out.write("\033[4A") # TODO: Support Windows
          @out.flush

          @recent_timing.delete_at(0) if @recent_timing.size > 100
        end

        def finalize
          @out.write("\033[J") # TODO: Support Windows
          @out.flush
        end
      end
    end
  end
end
