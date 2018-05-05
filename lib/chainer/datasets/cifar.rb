require 'datasets'

module Chainer
  module Datasets
    module CIFAR
      def self.get_cifar10(with_label: true, ndim: 3, scale: 1.0)
        get_cifar(10, with_label, ndim, scale)
      end

      def self.get_cifar100(with_label: true, ndim: 3, scale: 1.0)
        get_cifar(100, with_label, ndim, scale)
      end

      def self.get_cifar(n_classes, with_label, ndim, scale)
        train_data = []
        train_labels = []
        ::Datasets::CIFAR.new(n_classes: n_classes, type: :train).each do |record|
          train_data << record.pixels
          train_labels << (n_classes == 10 ? record.label : record.fine_label)
        end

        test_data = []
        test_labels = []
        ::Datasets::CIFAR.new(n_classes: n_classes, type: :test).each do |record|
          test_data << record.pixels
          test_labels << (n_classes == 10 ? record.label : record.fine_label)
        end

        [
          preprocess_cifar(Numo::UInt8[*train_data], Numo::UInt8[*train_labels], with_label, ndim, scale),
          preprocess_cifar(Numo::UInt8[*test_data], Numo::UInt8[*test_labels], with_label, ndim, scale)
        ]
      end

      def self.preprocess_cifar(images, labels, withlabel, ndim, scale)
        if ndim == 1
          images = images.reshape(images.shape[0], 3072)
        elsif ndim == 3
          images = images.reshape(images.shape[0], 3, 32, 32)
        else
          raise 'invalid ndim for CIFAR dataset'
        end
        images = images.cast_to(Numo::DFloat)
        images *= scale / 255.0

        if withlabel
          labels = labels.cast_to(Numo::Int32)
          TupleDataset.new(images, labels)
        else
          images
        end
      end
    end
  end
end

