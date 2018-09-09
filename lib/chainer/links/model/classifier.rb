module Chainer
  module Links
    module Model
      class Classifier < Chain
        attr_accessor :compute_accuracy
        attr_reader :predictor

        # @param [Chainer::Link] predictor Predictor network.
        # @param [Function] lossfun Loss function.
        # @param [Function] accfun Function that computes accuracy.
        # @param [Integer, String] label_key Key to specify label variable from arguments.
        #   When it is Integer, a variable in positional arguments is used.
        #   And when it is String, a variable in keyword arguments is used.
        def initialize(predictor, lossfun=Functions::Loss::SoftmaxCrossEntropy.method(:softmax_cross_entropy), accfun=Functions::Evaluation::Accuracy.method(:accuracy), label_key=-1)
          super()

          unless label_key.is_a?(Integer) || label_key.is_a?(String)
            raise TypeError, "label_key must be Integer or String, but is #{label_key.class}"
          end

          @lossfun = lossfun
          @accfun = accfun
          @y = nil
          @loss = nil
          @accuracy = nil
          @compute_accuracy = true
          @label_key = label_key

          init_scope do
            @predictor = predictor
          end
        end

        def call(*args, **kwargs)
          if @label_key.is_a?(Integer)
            raise IndexError, "label_key #{@label_key} is out of bounds" if @label_key < -args.size || @label_key >= args.size
            t = args.slice!(@label_key)
          elsif @label_key.is_a?(String)
            raise KeyError, "label_key #{@label_key} is not found" unless kwargs.has_key?(@label_key)
            t = kwargs[@label_key]
            kwargs.delete(@label_key)
          end

          @y = nil
          @accuracy = nil
          @y = @predictor.(*args, **kwargs)

          @loss = @lossfun.call(@y, t)
          Chainer::Reporter.save_report({loss: @loss}, self)
          if @compute_accuracy
            @accuracy = @accfun.call(@y, t)
            Chainer::Reporter.save_report({accuracy: @accuracy}, self)
          end
          @loss
        end
      end
    end
  end
end
