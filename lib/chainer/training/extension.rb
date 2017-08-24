module Chainer
  module Training
    class Extension
      PRIORITY_READER = 100

      attr_accessor :name

      def initialize
      end

      def call(trainer)
      end

      def default_name
        self.class.to_s
      end
    end
  end
end
