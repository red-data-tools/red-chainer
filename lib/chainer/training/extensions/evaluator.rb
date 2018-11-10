module Chainer
  module Training
    module Extensions
      # Trainer extension to evaluate models on a validation set.
      # This extension evaluates the current models by a given evaluation function.
      #
      # It creates a Chainer::Reporter object to store values observed in
      # the evaluation function on each iteration. The report for all iterations
      # are aggregated to Chainer::DictSummary. The collected mean values
      # are further reported to the reporter object of the trainer, where the name
      # of each observation is prefixed by the evaluator name. See
      # Chainer::Reporter for details in naming rules of the reports.
      #
      # Evaluator has a structure to customize similar to that of Chainer::Training::StandardUpdater.
      # The main differences are:
      #
      # - There are no optimizers in an evaluator. Instead, it holds links to evaluate.
      # - An evaluation loop function is used instead of an update function.
      # - Preparation routine can be customized, which is called before each evaluation.
      #   It can be used, e.g., to initialize the state of stateful recurrent networks.
      #
      # There are two ways to modify the evaluation behavior besides setting a custom evaluation function.
      # One is by setting a custom evaluation loop via  the `eval_func` argument.
      # The other is by inheriting this class and overriding the `evaluate` method.
      # In latter case, users have to create and handle a reporter object manually.
      # Users also have to copy the iterators before using them, in order to reuse them at the next time of evaluation.
      # In both cases, the functions are called in testing mode  (i.e., `chainer.config.train` is set to `false`).
      #
      # This extension is called at the end of each epoch by default.
      class Evaluator < Extension
        # @param [Dataset::Iterator] iterator Dataset iterator for the validation dataset. It can also be a dictionary of iterators.
        #                                     If this is just an iterator, the iterator is registered by the name 'main'.
        # @param [Chainer::Link] target Link object or a dictionary of links to evaluate.
        #                               If this is just a link object, the link is registered by the name 'main'.
        # @param [Dataset::Convert] converter Converter function to build input arrays.
        #             												`Chainer::Dataset.concat_examples` is used by default.
        # @param [Chainer::Device] device Device to which the training data is sent.
        # @param [Function] eval_hook Function to prepare for each evaluation process.
        #															It is called at the beginning of the evaluation.
        # 														The evaluator extension object is passed at each call.
        # @param [Function] eval_func Evaluation function called at each iteration.
        #                             The target link to evaluate as a callable is used by default.
        def initialize(iterator, target, converter: nil, device: nil, eval_hook: nil, eval_func: nil)
          @priority = Extension::PRIORITY_WRITER
          @trigger = [1, 'epoch']

          if iterator.kind_of?(Dataset::Iterator)
            iterator = { main: iterator }
          end
          @iterators = iterator

          if target.kind_of?(Link)
            target = { main: target }
          end
          @targets = target

          @converter = converter || Dataset::Convert.method(:concat_examples)
          @device = device
          @eval_hook = eval_hook
          @eval_func = eval_func
        end

        # Executes the evaluator extension.
        #
        # Unlike usual extensions, this extension can be executed without passing a trainer object.
        # This extension reports the performance on validation dataset using the `Chainer.report` function.
        # Thus, users can use  this extension independently from any trainer by manually configuring a `Chainer::Reporter` object.
        #
        # @param [Chainer::Training::Trainer] trainer Trainer object that invokes this extension.
        #                                     It can be omitted in case of calling this extension manually.
        def call(trainer = nil)
          reporter = Reporter.new
          prefix = self.respond_to?(:name) ? "#{self.name}/" : ""

          @targets.each do |name, target|
            reporter.add_observer("#{prefix}#{name}", target)
            reporter.add_observers("#{prefix}#{name}", target.method(:namedlinks), skipself: true)
          end

          result = nil
          reporter.scope(reporter.observation) do
            old_train = Chainer.configuration.train
            Chainer.configuration.train = false
            result = evaluate()
            Chainer.configuration.train = old_train
          end

          Reporter.save_report(result)
          return result
        end

        # Evaluates the model and returns a result dictionary.
        # This method runs the evaluation loop over the validation dataset.
        # It accumulates the reported values to `DictSummary` and returns a dictionary whose values are means computed by the summary.
        #
        # Users can override this method to customize the evaluation routine.
        # @return dict Result dictionary. This dictionary is further reported via `Chainer.save_report` without specifying any observer.
        def evaluate
          iterator = @iterators[:main]
          target = @targets[:main]
          eval_func = @eval_func || target

          @eval_hook.(self) if @eval_hook

          if iterator.respond_to?(:reset)
            iterator.reset
            it = iterator
          else
            it = iterator.dup
          end

          summary = DictSummary.new

          until it.is_new_epoch do
            batch = it.next
            observation = {}
            Reporter.report_scope(observation) do
              in_arrays = @converter.(batch, device: @device)

              old_enable_backprop = Chainer.configuration.enable_backprop
              Chainer.configuration.enable_backprop = false

              if in_arrays.kind_of?(Array)
                eval_func.(*in_arrays)
              elsif in_arrays.kind_of?(Hash)
                eval_func.(**in_arrays)
              else
                eval_func.(in_arrays)
              end

              Chainer.configuration.enable_backprop = old_enable_backprop
            end
            summary.add(observation)
          end

          summary.compute_mean()
        end

        def default_name
          "validation"
        end
      end
    end
  end
end
