require 'datasets'

module Chainer
  module Datasets
    module MNIST
      def self.get_mnist(withlabel: true, ndim: 1, scale: 1.0, dtype: nil, label_dtype: nil)
        xm = Chainer::Device.default.xm
        dtype ||= xm::SFloat
        label_dtype ||= xm::Int32

        train_raw = retrieve_mnist(type: :train)
        train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype, label_dtype)

        test_raw = retrieve_mnist(type: :test)
        test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype, label_dtype)
        [train, test]
      end

      def self.preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype)
        images = raw[:x]
        if ndim == 2
          images = images.reshape(true, 28, 28)
        elsif ndim == 3
          images = images.reshape(true, 1, 28, 28)
        elsif ndim != 1
          raise "invalid ndim for MNIST dataset"
        end

        images = images.cast_to(image_dtype)
        images *= scale / 255.0

        if withlabel
          labels = raw[:y].cast_to(label_dtype)
          TupleDataset.new(images, labels)
        else
          images
        end
      end

      def self.retrieve_mnist(type:)
        train_table = ::Datasets::MNIST.new(type: type).to_table

        xm = Chainer::Device.default.xm
        { x: xm::UInt8[*train_table[:pixels]], y: xm::UInt8[*train_table[:label]] }
      end
    end
  end
end
