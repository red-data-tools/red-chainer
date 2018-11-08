module Chainer
  module Serializers
    class MarshalSerializer < Chainer::Serializer
      attr_accessor :target, :path

      # @param [string] file_path Target file path
      # @param [Object] obj Object to be serialized
      def self.save_file(file_path, obj)
        s = self.new
        s.save(obj)
        File.open(file_path, 'wb') do |f|
          Marshal.dump(s.target, f)
        end
      end

      def initialize(target: nil, path: "", xm: Chainer.get_default_device.xm)
        @target = target.nil? ? {} : target
        @path = path
        @xm = xm
      end

      def [](key)
        self.class.new(target: @target, path: File.join(@path, key, '/'))
      end

      def call(key, value)
        ret = value
        if value.is_a?(TrueClass)
          arr = @xm::Bit[1]
        elsif value.is_a?(FalseClass)
          arr = @xm::Bit[0]
        elsif value.instance_of?(String) || value.nil?
          arr = value
        else
          arr = @xm::NArray.cast(value)
        end
        @target[File.join(@path, key)] = arr
        ret
      end
    end

    class MarshalDeserializer < Chainer::Deserializer
      # Loads an object from the file in Marshal format.
      # This is a short-cut function to load from an Marshal file that contains only one object.
      #
      # @param [string ]filename: Name of the file to be loaded.
      # @param [object] obj: Object to be deserialized. It must support serialization protocol.
      def self.load_file(filename, obj)
        File.open(filename) do |f|
          d = self.new(Marshal.load(f))
          d.load(obj)
        end
      end

      def initialize(marshalData, path: '', strict: true)
        @marshal_data = marshalData
        @path = path
        @strict = strict
      end

      def [](key)
        self.class.new(@marshal_data, path: File.join(@path, key, '/'), strict: @strict)
      end

      def call(key, value)
        key = File.join(@path, key)
        if !@strict && !@marshal_data.keys.include?(key)
          return value
        end

        dataset = @marshal_data[key]
        if value.nil?
          return dataset
        elsif value.instance_of?(String)
          return dataset
        elsif Chainer.array?(value)
          value.store(dataset)
          return value
        elsif value.is_a?(TrueClass) || value.is_a?(FalseClass)
          return dataset[0] == 1
        else
          return dataset[0]
        end
      end
    end
  end
end

