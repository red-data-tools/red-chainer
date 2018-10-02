module Chainer
  module Functions
    module Math
      # Sum of array elements over a given axis.
      class Sum < Chainer::FunctionNode
        # Sum of array elements over a given axis
        #
        # @param [Chainer::Variable] x Elements to sum
        # @param [nil, Integer, Array<Integer>] axis Axis which a sum is performed
        # @param[boolean] keepdims If `true`, the specified axes are remained as axes of length one
        # @return [Chainer::Variable] Output variable
        def self.sum(x, axis: nil, keepdims: false)
          y = Sum.new(axis: axis, keepdims: keepdims).apply([x]).first
          y
        end

        def initialize(axis: nil, keepdims: false)
          if axis.nil?
            @axis = nil
          elsif axis.is_a?(Integer)
            @axis = [axis]
          elsif axis.is_a?(Array) && axis.all? { |e| e.is_a?(Integer) }
            raise ArgumentError, "duplicate value in axis: #{axis}" unless axis.uniq.size == axis.size
            @axis = axis
          else
            raise TypeError, 'nil, Integer or Array of int are required'
          end

          @keepdims = keepdims
        end

        def forward(inputs)
          x = inputs.first
          ret = x.sum(axis: @axis, keepdims: @keepdims)
          ret = Numo::NArray.cast(ret)
          [ret]
        end

        def backward(indexes, grad_outputs)
          gy = grad_outputs.first
          ndim = @inputs.first.shape.size
          unless ndim == 0 || @axis.nil? || @keepdims
            actual_axis = @axis.map { |axis| axis >= 0 ? axis : axis + ndim  }
            shape = gy.shape
            actual_axis.sort.each { |axis| shape.insert(axis, 1) }
            gy = Chainer::Functions::Array::Reshape.reshape(gy, shape)
          end
          [Chainer::Functions::Array::BroadcastTo.broadcast_to(gy, @inputs.first.shape)]
        end
      end
    end
  end
end
