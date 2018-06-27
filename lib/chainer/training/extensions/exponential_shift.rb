module Chainer
  module Training
    module Extensions
      # Trainer extension to exponentially shift an optimizer attribute.
      #
      # This extension exponentially increases or decreases the specified attribute of the optimizer.
      # The typical use case is an exponential decay of the learning rate.
      # This extension is also called before the training loop starts by default.
      class ExponentialShift < Extension
        attr_reader :last_value

        # @param [string] attr Name of the attribute to shift
        # @param [float] rate Rate of the exponential shift.
        # @param [float] init Initial value of the attribute.
        # @param [float] target Target value of the attribute.
        # @param [Chainer::Optimizer] optimizer Target optimizer to adjust the attribute.
        def initialize(attr, rate, init: nil, target: nil, optimizer: nil)
          @attr = attr
          raise 'ExponentialShift does not support negative rate' if rate < 0
          @rate = rate
          @init = init
          @target = target
          @optimizer = optimizer
          @t = 0
          @last_value = nil
        end

        def init(trainer)
          optimizer = get_optimizer(trainer)
          @init = optimizer.send(@attr) if @init.nil?
          if @last_value.nil?
            update_value(optimizer, @init)
          else
            update_value(optimizer, @last_value)
          end
        end

        def call(trainer)
          @t += 1

          optimizer = get_optimizer(trainer)
          value = @init * (@rate ** @t)
          if @target
            if @rate > 1
              if value / @target > 1
                value = @target
              end
            else
              if value / @target < 1
                value = @target
              end
            end
          end
          update_value(optimizer, value)
        end

        def serialize(serializer)
          @t = serializer.('t', @t)
          @last_value = serializer.('last_value', @last_value)
          if @last_value.is_a?(Numo::NArray)
            @last_value = @last_value[0]
          end
        end

        private

        def get_optimizer(trainer)
          @optimizer || trainer.updater.get_optimizer(:main)
        end

        def update_value(optimizer, value)
          optimizer.send("#{@attr}=", value)
          @last_value = value
        end
      end
    end
  end
end
