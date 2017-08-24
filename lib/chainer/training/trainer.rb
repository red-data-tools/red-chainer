module Chainer
  module Training
    class ExtensionEntry
      def initialize(extension, priority, trigger, invoke_before_training)
        @extension = extension
        @trigger = trigger
        @invoke_before_training = invoke_before_training
        @priority = priority
      end
    end

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

      def extend(extension, name: nil, trigger: nil, priority: nil, invoke_before_training: nil)
        if name.nil?
          name = if extension.methods.include?(:name)
                   extension.name
                 elsif extension.methods.include?(:default_name)
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
    end
  end
end
