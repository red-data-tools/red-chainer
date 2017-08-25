module Chainer
  module Training
    class Extension
      PRIORITY_WRITER = 300
      PRIORITY_EDITOR = 200
      PRIORITY_READER = 100

      attr_accessor :name, :priority

      def initialize
      end

      def call(trainer)
      end

      def default_name
        self.class.to_s
      end

      def priority
        @priority || PRIORITY_READER
      end
    end
  end
end
