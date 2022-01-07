module Chainer
  module Links
    module Connection
      class Linear < ::Chainer::Link
        attr_reader :w, :b

        def initialize(in_size, out_size: nil, nobias: false, initial_w: nil, initial_bias: nil)
          super()
          in_size, out_size = nil, in_size if out_size.nil?
          @out_size = out_size

          init_scope do
            w_initializer = Chainer::Initializers.get_initializer(initial_w)
            @w = Chainer::Parameter.new(initializer: w_initializer)
            
            initialize_params(in_size) unless in_size.nil?
            
            if nobias
              @b = nil
            else
              initial_bias = 0 if initial_bias.nil?
              bias_initializer = Chainer::Initializers.get_initializer(initial_bias)
              @b = Chainer::Parameter.new(initializer: bias_initializer, shape: out_size)
            end
          end
        end

        def call(x)
          if @w.data.nil?
            initialize_params(x.size.div(x.shape[0]))
          end
          Chainer::Functions::Connection::LinearFunction.linear(x, @w, @b)
        end

        def initialize_params(in_size)
          @w.init([@out_size, in_size])
        end
      end
    end
  end
end
