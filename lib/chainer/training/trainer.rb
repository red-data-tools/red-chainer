module Chainer
  module Training
    class ExtensionEntry
      attr_accessor :extension, :trigger, :invoke_before_training, :priority

      def initialize(extension, priority, trigger, invoke_before_training)
        @extension = extension
        @trigger = trigger
        @invoke_before_training = invoke_before_training
        @priority = priority
      end
    end

    class Trainer
      attr_accessor :updater, :stop_trigger, :observation, :out

      def initialize(updater, stop_trigger: nil, out: 'result')
        @updater = updater
        @stop_trigger = Chainer::Training::Util.get_trigger(stop_trigger)
        @observation = {}
        @out = out

        reporter = Reporter.new
        updater.get_all_optimizers().each do |(name, optimizer)|
          reporter.add_observer(name, optimizer.target)
          optimizer.target.namedlinks(skipself: true) do |suffix, observer|
            observer_name = name.to_s + suffix
            reporter.add_observer(observer_name, observer)
          end
        end
        @reporter = reporter

        @done = false
        @extensions = {}

        @start_at = nil
        @snapshot_elapsed_time = 0.0
        @final_elapsed_time = nil

        updater.connect_trainer(self)
      end

      def elapsed_time
        return @final_elapsed_time if @done
        raise "training has not been started yet" if @start_at.nil?

        Time.now.to_f - @start_at + @snapshot_elapsed_time.to_f
      end

      def extend(extension, name: nil, trigger: nil, priority: nil, invoke_before_training: nil)
        if name.nil?
          name = if extension.name
                   extension.name
                 elsif extension.default_name
                   extension.default_name
                 else
                   raise ArgumentError 'name is not given for the extension'
                 end
        end

        raise 'the name "training" is prohibited as an extension name' if name == 'training'

        if trigger.nil?
          trigger = extension.methods.include?(:trigger) ? extension.trigger : [1, 'iteration']
        end
        trigger = Chainer::Training::Util.get_trigger(trigger)

        if priority.nil?
          priority = extension.methods.include?(:priority) ? extension.priority : Extension::PRIORITY_READER
        end

        if invoke_before_training.nil?
          invoke_before_training = extension.methods.include?(:invoke_before_training) ? extension.invoke_before_training : false
        end

        modified_name = name
        ordinal = 0

        @extensions.each do |modified_name|
          ordinal += 1
          modified_name = "#{name}_#{ordinal}"
        end

        extension.name = modified_name
        @extensions[modified_name] = ExtensionEntry.new(extension, priority, trigger, invoke_before_training)
      end

      def get_extension(name)
        if @extensions.keys.include?(name)
          @extensions[name].extension
        else
          raise "extension #{name} not found"
        end
      end

      def run
        raise 'cannot run training loop multiple times' if @done
        FileUtils.mkdir_p(@out)

        extensions = @extensions.sort_by { |(_, e)| -e.priority }.map { |(name, extension)| [name, extension] }

        @start_at = Time.now.to_f

        extensions.each do |(_, entry)|
          initializer = entry.extension.methods.include?(:init) ? entry.extension.method(:init) : nil
          initializer.call(self) if initializer
        end

        update = @updater.method(:update)
        reporter = @reporter
        stop_trigger = @stop_trigger

        begin
          until stop_trigger.(self) do
            @observation = {}
            reporter.scope(@observation) do
              update.call
              extensions.each do |(name, entry)|
                entry.extension.(self) if entry.trigger.(self)
              end
            end
          end
        ensure
          extensions.each do |(_, entry)|
            finalize = entry.extension.methods.include?(:finalize) ? entry.extension.method(:finalize) : nil
            finalize.() if finalize
          end
          @updater.finalize()
        end

        @final_elapsed_time = @elapsed_time
        @done = true
      end

      def serialize(serializer)
        updater.serialize(serializer['updater'])
        if @stop_trigger.respond_to?(:serialize)
          @stop_trigger.serialize(serializer['stop_trigger'])
        end

        s = serializer['extensions']
        t = serializer['extension_triggers']
        @extensions.each do |name, entry|
          if entry.extension.respond_to?(:serialize)
            entry.extension.serialize(s[name])
          end
          if entry.trigger.respond_to?(:serialize)
            entry.trigger.serialize(t[name])
          end
        end
        if serializer.is_a?(Chainer::Serializer)
          serializer.('_snapshot_elapsed_time', elapsed_time)
        else
          @snapshot_elapsed_time = serializer.('_snapshot_elapsed_time', 0.0)
        end
      end
    end
  end
end
