module Chainer
  module Training
    class StandardUpdater < Updater
      def initialize(iterator, optimizer, converter: nil, device: nil, loss_func: nil)
        if iterator.kind_of?(Dataset::Iterator)
          iterator = { main: iterator }
        end
        @iterators = iterator

        unless optimizer.kind_of?(Hash)
          optimizer = { main: optimizer }
        end
        @optimizers = optimizer

        @converter = converter || Dataset::Convert.method(:concat_examples)
        @loss_func = loss_func
        @device = device
        @iteration = 0
      end

      def get_all_optimizers
        @optimizer.to_h
      end
    end
  end
end
