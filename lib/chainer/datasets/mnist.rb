require 'zlib'

module Chainer
  module Datasets
    module Mnist
      def self.get_mnist(withlabel: true, ndim: 1, scale: 1.0, dtype: Numo::DFloat, label_dtype: Numo::Int32)
        train_raw = retrieve_mnist_training
        train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype, label_dtype)

        test_raw = retrieve_mnist_test
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

      def self.retrieve_mnist_training
        urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz']
        retrieve_mnist('train.npz', urls)
      end

      def self.retrieve_mnist_test
        urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
        retrieve_mnist('test.npz', urls)
      end

      def self.retrieve_mnist(name, urls)
        root = Chainer::Dataset::Download.get_dataset_directory('pfnet/chainer/mnist')
        path = File.expand_path(name, root)
        Chainer::Dataset::Download.cache_or_load_file(path) do
          make_npz(path, urls)
        end
      end

      def self.make_npz(path, urls)
        x_url, y_url = urls
        x_path = Chainer::Dataset::Download.cached_download(x_url)
        y_path = Chainer::Dataset::Download.cached_download(y_url)

        x = nil
        y = nil

        Zlib::GzipReader.open(x_path) do |fx|
          Zlib::GzipReader.open(y_path) do |fy|
            fx.read(4)
            fy.read(4)

            n = fx.read(4).unpack('i>')[0]
            fy.read(4)
            fx.read(8)

            x = Numo::UInt8.new(n, 784).rand(n)
            y = Numo::UInt8.new(n).rand(n)

            n.times do |i|
              y[i] = fy.read(1).ord
              784.times do |j|
                x[i, j] = fx.read(1).ord
              end
            end
          end
        end

        { x: x, y: y}
      end
    end
  end
end
