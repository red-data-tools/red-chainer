module Chainer::Functions::Array
  class SplitAxis < Chainer::Function
  # Function that splits multiple arrays along the specified axis.
    def initialize(indices_or_sections, axis)
      unless indices_or_sections.is_a?(Integer) || indices_or_sections.is_a?(Enumerable)
        raise TypeError.new('indices_or_sections must be integer or 1-D array')
      end

      @indices_or_sections = indices_or_sections
      @axis = axis
    end

    def forward(x)
      retain_inputs([])
      if @indices_or_sections.is_a?(Enumerable)
        ind = @indices_or_sections.to_a
        ind << x[0].shape[@axis]
      end
      @x_shape = x[0].shape
      @x_dtype = x[0].class
      x[0].split(@indices_or_sections, axis: @axis)
    end

    def backward(x, gys)
      if gys.any?(&:nil?)
        gx = @x_dtype.zeros(@x_shape)
        gxs = gx.split(@indices_or_sections, @axis)
        gxs.zip(gys).each do |gxi, gy|
          next unless gy
          gxi[[:*] * gxi.shape.size] = gy
        end
        [gx]
      else
        @x_dtype.concatenate(gys, axis: @axis)
      end
    end

    # Splits given variables along an axis.
    #
    # Args:
    #   x (tuple of Variables): Variables to be split.
    #   indices_or_sections (int or 1-D array): If this argument is an integer,
    #     N, the array will be divided into N equal arrays along axis.
    #     If it is a 1-D array of sorted integers, it
    #     indicates the positions where the array is split.
    #   axis (int): Axis that the input array is split along.
    #   force_tuple (bool): If ``True`` (the default) this method returns a
    #     tuple even when the number of outputs is one. Otherwise, if
    #     ``False`` a Variable will be returned when the number of outputs
    #     is one.
    #
    # Returns:
    #   tuple or Variable: Tuple of :class:`~chainer.Variable` objects
    #       if the number of outputs is more than 1 or
    #       :class:`~chainer.Variable` otherwise.
    #       When ``force_tuple`` is ``True``, returned value is always a tuple
    #       regardless of the number of outputs.
    #
    # .. note::
    #   This function raises :class:`ValueError` if at least
    #   one of the outputs is split to zero-size
    #   (i.e. ``axis``-th value of its shape is zero).
    def self.split_axis(x, indices_or_sections, axis, force_tuple: true)
      res = SplitAxis.new(indices_or_sections, axis).(x)
      if force_tuple and res.is_a?(Chainer::Variable)
        res = [res]
      end
      res
    end
  end
end
