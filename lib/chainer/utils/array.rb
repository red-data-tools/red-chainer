module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        if dtype == nil
          x.class.[](*x)
        else
          dtype.[](*x)
        end
      end
    end
  end
end
