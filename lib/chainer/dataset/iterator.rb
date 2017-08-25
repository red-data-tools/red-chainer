module Chainer
  module Dataset
    class Iterator
      def next
        raise NotImplementedError
      end

      def finalize
      end

      def serialize(serializer)
      end
    end
  end
end
