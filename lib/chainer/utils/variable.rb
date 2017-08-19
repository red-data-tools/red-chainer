module Chainer
  module Utils
    module Variable
      def self.check_grad_type(func, x, gx)
        if x.data.nil? || gx.nil?
          return
        end

        unless gx.instance_of?(x.data.class)
          raise TypeError, "Type of data and grad mismatch\n#{x.class} != #{gx.class}"
        end

        unless gx.shape == x.data.shape
          raise TypeError, "Shape of data and grad mismatch\n#{x.class} != #{gx.class}"
        end
      end
    end
  end
end

