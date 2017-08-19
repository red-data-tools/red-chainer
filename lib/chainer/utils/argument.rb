module Chainer
  module Utils
    module Argument
      def self.parse_kwargs(kwargs, **name_and_values)
        args = name_and_values.each_with_object({}) do |(key, val), h|
          h[key] = kwargs.delete(key) || val
        end

        unless kwargs.empty?
          raise TypeError, "got unexpected keyword argument(s) #{kwargs.keys().join(", ")}"
        end

        args
      end
    end
  end
end
