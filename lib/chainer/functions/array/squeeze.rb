module Chainer
  module Functions
    module Array
      class Squeeze < FunctionNode
        # Remove demensions of size one from the shape of a Numo::NArray.
        # @param [Chainer::Variable or Numo::NArray] x Input data.
        # @param [nil or integer or array of integer] axis A subset of the single-dimensional entries in the shape to remove.
        #   If `nil` is supplied, all of them are removed. The dimension index starts at zero.
        #   If an axis with dimension greater than one is selected, an error is raised.
        # @return [Chainer::Variable] Variable whose dimensions of size 1 are removed.
        def self.squeeze(x, axis: nil)
          self.new(axis: axis).apply([x]).first
        end

        def initialize(axis: nil)
          if axis.nil?
            @axis = nil
          elsif axis.kind_of?(Integer)
            @axis = [axis]
          elsif axis.kind_of?(::Array) && Array(axis).all? { |i| i.kind_of?(Integer) }
            @axis = axis
          else
            raise TypeError, 'axis must be None, int or tuple of ints'
          end
        end

        def forward(inputs)
          x = inputs.first
          shape = x.shape

          # TODO: numpy.squeeze
          if @axis.nil?
            new_shape = shape.reject { |axis| axis == 1 }
          else
            new_shape = shape
            @axis.map do |a|
              raise StandardError, "cannot select an axis to squeeze out which has size not equal to one" unless shape[a] == 1
              new_shape[a] = nil
            end
            new_shape.compact!
          end
          ret = new_shape.size.zero? ? x.class.new.fill(x[0]) : x.reshape(*new_shape)

          [ret]
        end

        def backward(indexes, grad_outputs)
          if @axis.nil?
            axis = argone(@inputs[0].shape)
          else
            axis = @axis
            ndim = @inputs[0].shape.size
            axis = axis.map { |x| x < 0 ? x + ndim : x }
            axis.sort!
          end
          gx = grad_outputs.first

          shape = gx.shape
          axis.each do |x|
            shape.insert(x, 1)
          end
          [gx.reshape(*shape)]
        end

        private

        def argone(iterable)
          result = []
          Array(iterable).each_with_index do |x, i|
            raise StandardError, "elements in iterable must be int" unless x.kind_of?(Integer)
            result << i if x == 1
          end
          result
        end
      end
    end
  end
end
