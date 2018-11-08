module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        if x.is_a? Integer or x.is_a? Float
          if dtype.nil?
            xm = Chainer.get_default_device.xm
            xm::NArray.cast(x)
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

      def self.rollaxis(y, axis, start: 0)
        axes = (0...y.ndim).to_a
        axes.delete_at(axis)
        axes.insert(start <= axes.size ? start : -1, axis)
        y.transpose(*axes)
      end

      def self.broadcast_to(x, shape)
        if x.shape.size > shape.size
           raise TypeError, "Shape of data  mismatch\n x.shape.size(#{x.shape.size}) > shape.size(#{shape.size})"
        end

        tile_shape = []
        shape[-x.shape.size..-1].each_with_index do |s, i|
          if  x.shape[i] == 1
            tile_shape << s
          elsif x.shape[i] == s
            tile_shape << 1
          else
            raise TypeError, "Shape of data  mismatch\n#{x.shape} != #{shape}"
          end
        end

        x.tile(*shape[0...-x.shape.size], *tile_shape)
      end
    end
  end
end
