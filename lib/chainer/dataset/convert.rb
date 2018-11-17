module Chainer
  module Dataset
    module Convert
      def self.to_device(device, x)
        # TODO(sonots): Implement after Cumo supports transferring between devices
        x
      end

      def self.concat_examples(batch, device: nil, padding: nil)
        raise "batch is empty" if batch.size == 0
        device = device ? Chainer.get_device(device) : Chainer.get_default_device # takes care of int and nil
        first_elem = batch[0]

        if first_elem.kind_of?(Array)
          result = []
          unless padding.kind_of?(Array)
            padding = [padding] * first_elem.size
          end

          first_elem.size.times do |i|
            x = _concat_arrays(batch.map { |b| b[i] }, padding[i], device)
            result.push(to_device(device, x))
          end

          return result
        else
          return _concat_arrays(batch, padding, device)
        end
      end

      def self._concat_arrays(arrays, padding, device)
        xm = device.xm
        unless arrays[0].kind_of?(xm::NArray)
          # [1, 2, 3, 4] => Numo::Int32[1, 2, 3, 4]
          arrays = xm::NArray.cast(arrays)
          if padding
            return _concat_arrays_with_padding(arrays, padding, device)
          end
          return arrays
        end

        if padding
          return _concat_arrays_with_padding(arrays, padding, device)
        end

        # [Numo::SFloat[1, 2], Numo::SFloat[3, 4]]
        #  => Numo::SFloat#shape=[2,2]
        # [[1, 2], [3, 4]]
        a = arrays.map{|arr| arr[:-, false]}
        a[0].concatenate(*a[1..-1])
      end

      def self._concat_arrays_with_padding(arrays, padding, device)
        xm = device.xm
        if Chainer.array?(arrays[0]) and arrays[0].ndim > 0
          xm = Chainer.get_array_module(arrays[0])
          shape = xm::Int32.cast(arrays[0].shape)
          arrays[1..-1].each do |array|
            if xm::Bit.[](shape != array.shape).any?
              shape = xm::Int32.maximum(shape, array.shape)
            end
          end
        else # Integer
          shape = []
        end

        shape = shape.insert(0, arrays.size).to_a
        if Chainer.array?(arrays[0]) and arrays[0].ndim > 0
          result = arrays[0].class.new(shape).fill(padding)
        else # Integer
          result = xm::Int32.new(shape).fill(padding)
        end

        arrays.size.times do |i|
          src = arrays[i]
          if Chainer.array?(src) and src.ndim > 0
            result[i, 0...src.shape[0], 0...src.shape[1]] = src
          else # Integer
            result[i] = src
          end
        end

        result
      end
    end
  end
end
