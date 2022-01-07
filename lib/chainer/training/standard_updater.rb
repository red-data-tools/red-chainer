module Chainer
  module Training
    class StandardUpdater < Updater
      attr_accessor :iteration

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

      def get_optimizer(name)
        @optimizers[name]
      end

      def get_all_optimizers
        @optimizers.to_h
      end

      def get_iterator(name)
        @iterators[name]
      end

      def update
        update_core
        @iteration += 1
      end

      def epoch
        @iterators[:main].epoch
      end

      def epoch_detail
        @iterators[:main].epoch_detail
      end

      def update_core
        batch = @iterators[:main].next
        in_arrays = @converter.call(batch, device: @device)

        optimizer = @optimizers[:main]
        loss_func = @loss_func || optimizer.target

        if in_arrays.kind_of?(Array)
          optimizer.update(loss_func, *in_arrays)
        elsif in_arrays.kind_of?(Hash)
          optimizer.update(loss_func, **in_arrays)
        else
          optimizer.update(loss_func, in_arrays)
        end
      end

      def finalize
        @iterators.each do |(_, iterator)|
          iterator.finalize
        end
      end

      def serialize(serializer)
        @iterators.each do |name, iterator|
          iterator.serialize(serializer["iterator:#{name}"])
        end
        @optimizers.each do |name, optimizer|
          optimizer.serialize(serializer["optimizer:#{name}"])
          optimizer.target.serialize(serializer["model:#{name}"])
        end

        @iteration = serializer.('iteration', @iteration)
      end
    end
  end
end
