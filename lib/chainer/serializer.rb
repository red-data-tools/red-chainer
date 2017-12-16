module Chainer
  # Abstract base class of all serializers and deserializers.
  class AbstractSerializer
    # Gets a child serializer.
    # This operator creates a child serializer represented by the given key.
    # 
    # @param [string] key: Name of the child serializer.
    def [](key)
      raise NotImplementedError
    end

    # Serializes or deserializes a value by given name.
    # This operator saves or loads a value by given name.
    # If this is a serializer, then the value is simply saved at the key.
    # Note that some type information might be missed depending on the
    # implementation (and the target file format).
    # If this is a deserializer, then the value is loaded by the key.
    # The deserialization differently works on scalars and arrays.
    # For scalars, the ``value`` argument is used just for determining the type of
    # restored value to be converted, and the converted value is returned.
    # For arrays, the restored elements are directly copied into the
    # ``value`` argument. String values are treated like scalars.
    #
    # @param [string] key: Name of the serialization entry.   
    # @param [any] value: Object to be (de)serialized.
    #                     ``None`` is only supported by deserializers.
    # @return Serialized or deserialized value.  
    def call(key, value)
      raise NotImplementedError  
    end
  end

  # Base class of all serializers.
  class Serializer < AbstractSerializer
    # Saves an object by this serializer.
    # This is equivalent to ``obj.serialize(self)``.
    #
    # @param [any] obj: Target object to be serialized.  
    def save(obj)
      obj.serialize(self)
    end
  end

  # Base class of all deserializers.
  class Deserializer < AbstractSerializer
    def load(obj)
      obj.serialize(self)
    end
  end
end
