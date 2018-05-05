module Chainer
  module Utils
    module Variable
      def self.check_grad_type(func, x, gx)
        if x.data.nil? || gx.nil?
          return
        end

        unless gx.is_a?(x.data.class.superclass)
          raise TypeError, "Type of data and grad mismatch\n#{x.data.class} != #{gx.class}"
        end

        unless gx.class == x.data.class
          raise TypeError, "Dtype(Class) of data and grad mismatch\n#{x.data.class} != #{gx.class}"
        end

        unless gx.shape == x.data.shape
          raise TypeError, "Shape of data and grad mismatch\n#{x.data.shape} != #{gx.shape}"
        end
      end
    end
  end
end

