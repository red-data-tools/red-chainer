module Chainer
  module Training
    class Updater
      def connect_trainer(trainer)
      end

      def finalize
      end

      def get_optimizer(name)
        raise NotImplementedError
      end

      def get_all_optimizers
        raise NotImplementedError
      end

      def update
        raise NotImplementedError
      end

      def serialize(serializer)
        raise NotImplementedError
      end
    end
  end
end
