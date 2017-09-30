module Chainer
  module Iterators
    class SerialIterator < Chainer::Dataset::Iterator 
      def initialize(dataset, batch_size, repeat: true, shuffle: true)
        @dataset = dataset
        @batch_size = batch_size
        @repeat = repeat
        @shuffle = shuffle

        reset
      end

      def next
        raise StopIteration if !@repeat && @epoch > 0

        @previous_epoch_detail = @epoch_detail

        i = @current_position
        i_end = i + @batch_size
        n = @dataset.size

        batch = @order[i...i_end].to_a.map { |index| @dataset[index] }

        if i_end >= n
          if @repeat
            rest = i_end - n
            unless @order.nil?
              @order = @order.class[*@order.to_a.shuffle]
            end
            if rest > 0
              if @order.nil?
                batch = batch.append(@dataset[0...rest])
              else
                batch = @dataset[0...rest].map { |index| @dataset[index] }
              end
            end
            @current_position = rest
          else
            @current_position = 0
          end

          @epoch += 1
          puts "epoch is #{@epoch}"
          @is_new_epoch = true
        else
          @is_new_epoch = false
          @current_position = i_end
        end

        batch
      end

      def epoch_detail
        @epoch + @current_position / @dataset.size
      end

      def reset
        if @shuffle
          order = @dataset.size.times.map(&:to_i).shuffle
          @order = Numo::Int64[*order]
        else
          order = @dataset.size.times.map(&:to_i)
          @order = Numo::Int64[*order]
        end

        @current_position = 0
        @epoch = 0
        @is_new_epoch = false
        @previous_epoch_detail = -1.0
      end
    end
  end
end
