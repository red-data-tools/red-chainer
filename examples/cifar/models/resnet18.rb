module ResNet18
  class Plain < Chainer::Chain
    include Chainer::Functions::Activation
    include Chainer::Initializers
    include Chainer::Links::Connection
    include Chainer::Links::Normalization

    def initialize(ch, stride, use_conv: false)
      super()

      @use_conv = use_conv
      w = HeNormal.new

      init_scope do
        @conv1 = Convolution2D.new(nil, ch, 3, stride: stride, pad: 1, nobias: true, initial_w: w)
        @bn1 = BatchNormalization.new(ch)
        @conv2 = Convolution2D.new(nil, ch, 3, stride: 1, pad: 1, nobias: true, initial_w: w)
        @bn2 = BatchNormalization.new(ch)
        if @use_conv
          @conv3 = Convolution2D.new(nil, ch, 3, stride: stride, pad: 1, nobias: true, initial_w: w)
          @bn3 = BatchNormalization.new(ch)
        end
      end
    end

    def call(x)
      h = Relu.relu(@bn1.(@conv1.(x)))
      h = @bn2.(@conv2.(h))
      if @use_conv
        h2 = @bn3.(@conv3.(x))
        Relu.relu(h + h2)
      else
        Relu.relu(h + x)
      end
    end
  end

  class Block < Chainer::ChainList
    def initialize(layer, ch, stride=2)
      super()
      add_link(Plain.new(ch, stride, use_conv: true))
      (layer-1).times do
        add_link(Plain.new(ch, 1))
      end
    end

    def call(x)
      @children.each do |f|
        x = f.(x)
      end
      x
    end
  end

  class Model < Chainer::Chain
    include Chainer::Functions::Activation
    include Chainer::Functions::Evaluation
    include Chainer::Functions::Loss
    include Chainer::Functions::Pooling
    include Chainer::Initializers
    include Chainer::Links::Connection
    include Chainer::Links::Normalization

    def initialize(n_classes: 10)
      super()
      initial_w = HeNormal.new

      init_scope do
        @conv = Convolution2D.new(3, 64, 7, stride: 2, pad: 3, initial_w: initial_w)
        @bn = BatchNormalization.new(64)

        @res2 = Block.new(2, 64, 1)
        @res3 = Block.new(2, 128)
        @res4 = Block.new(2, 256)
        @res5 = Block.new(2, 512)
        @fc = Linear.new(nil, out_size: n_classes)
      end
    end

    def call(x)
      h = Relu.relu(@bn.(@conv.(x)))
      h = @res2.(h)
      h = @res3.(h)
      h = @res4.(h)
      h = @res5.(h)
      h = AveragePooling2D.average_pooling_2d(h, h.shape[2..-1])
      @fc.(h)
    end
  end
end
