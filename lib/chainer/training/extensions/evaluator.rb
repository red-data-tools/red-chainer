module Chainer
  module Training
    module Extensions
      class Evaluator < Extension
        def initialize(iterator, target, converter: nil, device: nil, eval_hook: nil, eval_func: nil)
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
      end
    end
  end
end
