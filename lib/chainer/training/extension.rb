module Chainer
  module Training
    class Extension
      PRIORITY_WRITER = 300
      PRIORITY_EDITOR = 200
      PRIORITY_READER = 100

      attr_accessor :name
      attr_writer :trigger, :priority

      def initialize
      end

      def call(trainer)
      end

      def default_name
        self.class.name.split('::').last
      end

      def trigger
        @trigger || [1, 'iteration']
      end

      def priority
        @priority || PRIORITY_READER
      end
    end
  end
end
