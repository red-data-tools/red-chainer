module Chainer
  module Datasets
    class TupleDataset
      def initialize(*datasets)
        if datasets.empty?
          raise "no datasets are given"
        end
        length = datasets[0].shape[0]

        datasets.each_with_index do |dataset, idx|
          raise "dataset of the index #{idx} has a wrong length" unless dataset.shape[0] == length
        end

        @datasets = datasets
        @length = length
      end

      def [](index)
        batches = @datasets.map { |dataset| dataset[index] }
        if index.kind_of?(Enumerable)
          length = batches[0].shape[0]
          length.times.map {|i| batches.map { |m| m[i] } }
        else
          batches
        end
      end

      def size
        @length
      end
    end
  end
end
