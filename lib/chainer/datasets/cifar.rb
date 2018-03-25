require 'rubygems/package'

module Chainer
  module Datasets
    module Cifar
      def self.get_cifar10(withlabel: true, ndim: 3, scale: 1.0)
        get_cifar('cifar-10', withlabel, ndim, scale)
      end

      def self.get_cifar100(withlabel: true, ndim: 3, scale: 1.0)
        get_cifar('cifar-100', withlabel, ndim, scale)
      end

      def self.get_cifar(name, withlabel, ndim, scale)
        root = Chainer::Dataset::Download.get_dataset_directory('cifar')
        filename = "#{name}-binary.tar.gz"
        url = "http://www.cs.toronto.edu/~kriz/#{filename}"
        path = File.expand_path(filename, root)
        extractfile(root, url)
        raw = creator(root, name)
        train = preprocess_cifar(raw[:train_x], raw[:train_y], withlabel, ndim, scale)
        test = preprocess_cifar(raw[:test_x], raw[:test_y], withlabel, ndim, scale)
        [train, test]
      end

      def self.extractfile(root, url)
        archive_path = Chainer::Dataset::Download.cached_download(url)
        Gem::Package::TarReader.new(Zlib::GzipReader.open(archive_path)) do |tar|
          tar.each do |entry|
            dest = File.expand_path(entry.full_name, root)
            if entry.directory?
              FileUtils.mkdir_p(dest)
            else
              File.open(dest, "wb") do |f|
                  f.print(entry.read)
              end
            end
          end
        end          
      end

      def self.creator(root, name)
        if name == 'cifar-10'
          train_x = Numo::UInt8.new(5, 10000, 3072).rand(1)
          train_y = Numo::UInt8.new(5, 10000).rand(1)
          test_x = Numo::UInt8.new(10000, 3072).rand(1)
          test_y = Numo::UInt8.new(10000).rand(1)
        
          dir = File.expand_path("cifar-10-batches-bin", root)
          (1..5).each do |i|
            file_name = "#{dir}/data_batch_#{i}.bin"
            open(file_name, "rb") do |f|
              s = 0
              while b = f.read(3073) do
                datasets = b.unpack("C*") 
                train_y[i - 1, s] = datasets.shift
                train_x[i - 1, s, false] = datasets
                s += 1
              end
            end
          end

          file_name = "#{dir}/test_batch.bin"
          open(file_name, "rb") do |f| 
            s = 0
            while b = f.read(3073) do
              datasets = b.unpack("C*") 
              test_y[s] = datasets.shift
              test_x[s, false] = datasets
              s += 1
            end
          end
          
          train_x = train_x.reshape(50000, 3072)
          train_y = train_y.reshape(50000)
        else
          train_x = Numo::UInt8.new(50017, 3072).rand(1)
          train_y = Numo::UInt8.new(50017).rand(1)
          test_x = Numo::UInt8.new(10004, 3072).rand(1)
          test_y = Numo::UInt8.new(10004).rand(1)
          dir = File.expand_path("cifar-100-binary", root)
          
          file_name = "#{dir}/train.bin"
          open(file_name, "rb") do |f| 
            s = 0
            while b = f.read(3073) do
              datasets = b.unpack("C*") 
              train_y[s] = datasets.shift
              train_x[s, false] = datasets
              s += 1
            end
          end
          
          file_name = "#{dir}/test.bin"
          open(file_name, "rb") do |f| 
            s = 0
            while b = f.read(3073) do
              datasets = b.unpack("C*") 
              test_y[s] = datasets.shift
              test_x[s, false] = datasets
              s += 1
            end
          end
        end

        {
          train_x: train_x,
          train_y: train_y,
          test_x: test_x,
          test_y: test_y
        }
      end

      def self.preprocess_cifar(images, labels, withlabel, ndim, scale)
        if ndim == 1
          images = images.reshape(images.shape[0], 3072)
        elsif ndim == 3
          images = images.reshape(images.shape[0], 3, 32, 32)
        else
          raise 'invalid ndim for CIFAR dataset'
        end
        images = images.cast_to(Numo::Float32)
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

