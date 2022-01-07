module Chainer::Functions::Array
  # Concatenate multiple tensors towards specified axis.
  class Concat < Chainer::Function
    # concat along the channel dimension by default
    def initialize(axis: 1)
      raise TypeError.new('axis must be Integer') unless axis.is_a?(Integer)
      @axis = axis
    end

    def forward(xs)
      retain_inputs(())
      @x_shapes = xs.map(&:shape)
      [xs[0].class.concatenate(xs, axis: @axis)]
    end


    def backward(xs, gy)
      return gy if xs.size == 1

      sizes = Numo::Int32[*@x_shapes.map{ |shape| shape[@axis] }].cumsum()
      gy[0].split(sizes, axis: @axis)
    end

    def self.concat(xs, axis: 1)
      # Concatenates given variables along an axis.

      # Args:
      #     xs (tuple of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
      #         :class:`cupy.ndarray`):
      #         Input variables to be concatenated. The variables must have the \
      #         same shape, except in the dimension corresponding to axis.
      #         axis (int): The axis along which the arrays will be joined. Default \
      #         is 1.

      # Returns:
      #     ~chainer.Variable: The concatenated variable.

      # .. admonition:: Example

      #     >>> x = np.arange(0, 12).reshape(3, 4)
      #     >>> x
      #     array([[ 0,  1,  2,  3],
      #     [ 4,  5,  6,  7],
      #     [ 8,  9, 10, 11]])
      #     >>> y = np.arange(0, 3).reshape(3, 1)
      #     >>> y
      #     array([[0],
      #            [1],
      #            [2]])
      #     >>> z = F.concat((x, y), axis=1)
      #     >>> z.data
      #     array([[ 0,  1,  2,  3,  0],
      #            [ 4,  5,  6,  7,  1],
      #            [ 8,  9, 10, 11,  2]])

      Concat.new(axis: axis).(*xs)
    end
  end
end
