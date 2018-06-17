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
        train_table = ::Datasets::CIFAR.new(n_classes: n_classes, type: :train).to_table
        test_table = ::Datasets::CIFAR.new(n_classes: n_classes, type: :test).to_table

        train_data = train_table[:pixels]
        test_data = test_table[:pixels]
        if n_classes == 10
          train_labels = train_table[:label]
          test_labels = test_table[:label]
        else
          train_labels = train_table[:fine_label]
          test_labels = test_table[:fine_label]
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
        images = images.cast_to(Numo::SFloat)
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

