module Chainer
  module Dataset
    module Convert
      def self.to_device(device, x)
        # TODO: support cuda
        x
      end

      def self.concat_examples(batch, device: nil, padding: nil)
        raise "batch is empty" if batch.size == 0
        first_elem = batch[0]

        if first_elem.kind_of?(Array)
          result = []
          unless padding.kind_of?(Array)
            padding = [padding] * first_elem.size
          end

          first_elem.size.times do |i|
            x = concat_arrays(batch.map { |b| b[i] }, padding[i])
            result.append(to_device(device, x))
          end
        end
      end

      def self.concat_arrays(arrays, padding)
        unless arrays[0].kind_of?(Numo::NArray)
          arrays = Numo::NArray.cast(arrays)
        end

        if padding
          return concat_arrays_with_padding(arrays, padding)
        end

        Numo::NArray.concatenate(arrays.map { |arr| arr[nil] })
      end

      def self.concat_arrays_with_padding(arrays, padding)
        shape = Numo::Int32.[](arrays[0].shape)
        arrays[1...arrays.len].each do |array|
          if Numo::Bit.[](shape != array.shape).any?
            # TODO: numpy maximum
          end
        end

        shape = [shape.insert(0, arrays.size)]
        result = arrays[0].dtype.[](*shape).full(padding)
        arrays.size.times do |i|
          src = arrays[i]
          slices = src.shape.map { |s| [s] }
          result[[i] + slices] = src
        end

        result
      end
    end
  end
end
