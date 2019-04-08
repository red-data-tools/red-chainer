module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        if x.is_a? Integer or x.is_a? Float
          if dtype.nil?
            xm = Chainer::Device.default.xm
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

      def self.ndindex(shape)
        shape.reduce(&:*).times.map do |i|
          shape.size.times.reduce([]) do |ndidx, j|
            ndidx << (i / shape.drop(j + 1).reduce(1, &:*)) % shape[j]
          end
        end
      end

      def self.take(x, indices, axis: nil)
        if axis
          dimensional_indices = ::Array.new(x.shape.size, true)

          indices_narray = Numo::Int32.cast(indices)
          if indices_narray.shape.size > 1
            y = x.class.zeros(*indices_narray.shape, *x.shape.drop(axis + 1))
            self.ndindex(indices_narray.shape).each do |ndidx|
              dimensional_indices[axis] = indices_narray[*ndidx]
              y[*ndidx, *::Array.new(x.shape.size - axis - 1, true)] = x[*dimensional_indices]
            end
            return y
          else
            dimensional_indices[axis] = indices
          end
          x[*dimensional_indices]
        else
          x[indices]
        end
      end

      def self.rollaxis(y, axis, start: 0)
        n = y.ndim
        # normalize axis
        axis = axis < 0 ? n + axis : axis
        if axis >= n
          raise ArgumentError, "axis #{axis} is out of bounds for array of dimension #{n}"
        end

        if start < 0
          start += n
        end

        unless 0 <= start && start < n + 1
          raise ArgumentError, "start arg requires #{-n} <= start < #{n}, but #{start} was passed in"
        end

        if axis < start
          start -= 1
        end

        if axis == start
          return y
        end

        axes = (0...n).to_a
        axes.delete_at(axis)
        axes.insert(start <= axes.size ? start : -1, axis)
        y.transpose(*axes)
      end

      def self.broadcast_to(x, shape)
        if x.shape.size > shape.size
           raise TypeError, "Shape of data  mismatch\n x.shape.size(#{x.shape.size}) > shape.size(#{shape.size})"
        end

        tile_shape = []
        if x.shape.size > 0
          shape[-x.shape.size..-1].each_with_index do |s, i|
            if  x.shape[i] == 1
              tile_shape << s
            elsif x.shape[i] == s
              tile_shape << 1
            else
              raise TypeError, "Shape of data  mismatch\n#{x.shape} != #{shape}"
            end
          end
        else
          tile_shape = shape
        end

        x.tile(*shape[0...-x.shape.size], *tile_shape)
      end
    end
  end
end
