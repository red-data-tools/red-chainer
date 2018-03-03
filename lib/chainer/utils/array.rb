module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        if dtype.nil?
          x.class.cast(x.dup)
        else
          dtype.cast(x.dup)
        end
      end
    end
  end
end
