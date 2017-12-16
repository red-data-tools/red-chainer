module Chainer
  module Serializers
    class MarshalSerializer < Chainer::Serializer      
      attr_accessor :target, :path

      def self.save_file(filename, obj)
        s = self.new
        s.save(obj)
        Marshal.dump(s.target, filename)
      end

      def initialize(target: nil, path: "")
        @target = target.nil? ? {} : target
        @path = path
      end

      def [](key)
        self.class.new(target: @target, path: File.join(@path, key, '/'))
      end

      def call(key, value)
        ret = value
        if value.is_a?(TrueClass)
          arr = Numo::Bit[1]
        elsif value.is_a?(FalseClass)
          arr = Numo::Bit[0]
        else
          arr = Numo::NArray.cast(value)
        end
        @target[File.join(@path, key)] = arr
        ret
      end
    end
  end
end

