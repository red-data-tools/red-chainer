require 'tempfile'
require 'json'

module Chainer
  module Training
    module Extensions
      class LogReport < Extension
        attr_reader :log

        def initialize(keys: nil, trigger: [1, 'epoch'], postprocess: nil, log_name: 'log')
          @keys = keys
          @trigger = Chainer::Training::Util.get_trigger(trigger)
          @postprocess = postprocess
          @log_name = log_name
          @log = []

          init_summary
        end

        def call(trainer)
          observation = trainer.observation

          if @keys.nil?
            @summary.add(observation)
          else
            symbolized_observation = Hash[observation.map{|(k,v)| [k.to_sym,v]}]
            filterd_keys = @keys.select {|k| observation.keys.include?(k.to_sym) }
            @summary.add(filterd_keys.each_with_object({}) {|k, hash| hash[k.to_s] = observation[k.to_sym] })            
          end

          # if trigger is true, output the result
          return unless @trigger.(trainer)

          stats = @summary.compute_mean
          stats_cpu = {}
          stats.each do |name, value|
            stats_cpu[name] = value.to_f  # copy to CPU
          end

          updater = trainer.updater
          stats_cpu['epoch'] = updater.epoch
          stats_cpu['iteration'] = updater.iteration
          stats_cpu['elapsed_time'] = trainer.elapsed_time
        
          @postprocess.(stats_cpu) unless @postprocess.nil?
          
          @log << stats_cpu

          unless @log_name.nil?
            # example: sprintf("%{a}, %{b}", {a: "1", b: "2"})
            # => "1, 2"
            log_name = sprintf(@log_name, stats_cpu)
            temp_file = Tempfile.create(basename: log_name, tmpdir: trainer.out)

            JSON.dump(@log, temp_file)

            new_path = File.join(trainer.out, log_name)
            FileUtils.mv(temp_file.path, new_path)
          end

          init_summary
        end

        def serialize(serializer)
          if @trigger.respond_to?(:serialize)
            @trigger.serialize(serializer['_trigger'])
          end
          # Note that this serialization may lose some information of small
          # numerical differences.
          if serializer.is_a?(Chainer::Serializer)
            log = JSON.generate(@log)
            serializer.('_log', log)
          else
            log = serializer.('_log', '')
            @log = JSON.parse(log)
          end
        end

        private

        def init_summary
          @summary = DictSummary.new
        end
      end
    end
  end
end
