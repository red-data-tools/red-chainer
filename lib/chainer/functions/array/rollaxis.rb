module Chainer
  module Functions
    module Array
      # Roll axis of an array.
      class Rollaxis < FunctionNode
        # Roll the axis backwards to the given position.
        #
        # @param [Chainer::Variable] x Input variable
        # @param [Integer] axis The axis to roll backwards.
        # @param [Integer] start The place to which the axis is moved.
        # @return [Chainer::Variable] Variable whose axis is rolled.
        def self.rollaxis(x, axis, start: 0)
          Rollaxis.new(axis, start).apply([x]).first
        end

        def initialize(axis, start)
          unless axis.is_a?(Integer)
            raise ArgumentError, 'axis must be int'
          end

          unless start.is_a?(Integer)
            raise ArgumentError, 'start must be int'
          end

          @axis = axis
          @start = start
        end

        def forward(inputs)
          retain_inputs([])
          @in_ndim = inputs.first.ndim

          [Chainer::Utils::Array.rollaxis(inputs.first, @axis, start: @start)]
        end

        def backward(indexes, gy)
          axis = @axis
          if axis < 0
            axis += @in_ndim
          end
          start = @start
          if start < 0
            start += @in_ndim
          end

          if axis > start
            axis += 1
          else
            start -= 1
          end

          Rollaxis.new(start, axis).apply(gy)
        end
      end
    end
  end
end
