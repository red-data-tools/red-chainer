module Chainer
  module Links
    module Connection
      class EmbedID < ::Chainer::Link
        attr_reader :w

        def initialize(in_size, out_size, initial_w: nil, ignore_label: nil)
          super()
          @ignore_label = ignore_label

          init_scope do
            initial_w ||= Chainer::Initializers::Normal.new(scale: 1.0)
            @w = Chainer::Parameter.new(initializer: initial_w, shape: [in_size, out_size])
          end
        end

        def call(x)
          Chainer::Functions::Connection::EmbedIDFunction.embed_id(x, @w, ignore_label: @ignore_label)
        end
      end
    end
  end
end
