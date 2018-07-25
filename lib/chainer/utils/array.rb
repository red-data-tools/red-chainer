module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        if x.is_a? Integer or x.is_a? Float
          if dtype.nil?
            Numo::NArray.cast(x)
          else
            dtype.cast(x.dup)
          end
        else
          if dtype.nil?
            x
          else
            dtype.cast(x)
          end
        end
      end

      def self.take(x, indices, axis: nil)
        if axis
          indices = make_indecies_with_axis(x.shape, indices, axis)
        end
        x[indices]
      end

      def self.make_indecies_with_axis(shape, indices, axis, values = [])
        target_axis = values.size
        if shape.size == values.size
          values.zip(shape.drop(1) + [1]).reduce(0) do |sum, (x, ndim)|
            (sum + x) * ndim
          end
        else
          enum = (axis == target_axis) ? indices : (0...shape[target_axis])
          if enum.is_a?(Integer)
            make_indecies_with_axis(shape, indices, axis, values + [indices])
          else
            enum.map do |x|
              make_indecies_with_axis(shape, indices, axis, values + [x])
            end
          end
        end
      end
    end
  end
end
