require "open-uri"
require "pstore"

module Chainer
  module Dataset
    module Download
      DATASET_ROOT = ENV.fetch("RED_CHAINER_DATASET_ROOT", File.expand_path(".chainer/dataset", "~"))

      def self.cached_download(url)
        cache_root = File.expand_path('_dl_cache', DATASET_ROOT)
        FileUtils.mkdir_p(cache_root)
        lock_path = File.expand_path('_dl_lock', cache_root)
        urlhash = Digest::MD5.hexdigest(url)
        cache_path = File.expand_path(urlhash, cache_root)

        return cache_path if File.exist?(cache_path)

        temp_root = Dir.mktmpdir(nil, cache_root)
        temp_path = File.expand_path('dl', temp_root)
        open(url) do |f|
          puts "Downloading from #{url}"
          open(temp_path, "w+b") do |out|
            out.write(f.read)
          end
          FileUtils.mv(temp_path, cache_path)
          FileUtils.rm_r(temp_root)
        end
        cache_path
      end

      def self.get_dataset_directory(dataset_name, create_directory: true)
        path = File.expand_path(dataset_name, DATASET_ROOT)
        FileUtils.mkdir_p(path) if create_directory
        path
      end

      def self.cache_or_load_file(path, data)
        return PStore.new(path).transaction { |t| t['data'] } if File.exist?(path)

        pstore = PStore.new(path)
        pstore.transaction{|t|
          t["data"] = data
        }

        data
      end
    end
  end
end
