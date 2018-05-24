class Block < Chainer::Chain
  def initialize(out_channels, ksize, pad: 1)
    super()
    init_scope do
      @conv = Chainer::Links::Connection::Convolution2D.new(nil, out_channels, ksize, pad: pad, nobias: true)
      @bn = Chainer::Links::Normalization::BatchNormalization.new(out_channels)
    end
  end

  def call(x)
    h = @conv.(x)
    h = @bn.(h)
    Chainer::Functions::Activation::Relu.relu(h)
  end
end

class VGG < Chainer::Chain
  def initialize(n_classes: 10)
    super()
    init_scope do
      @block1_1 = Block.new(64, 3)
      @block1_2 = Block.new(64, 3)
      @block2_1 = Block.new(128, 3)
      @block2_2 = Block.new(128, 3)
      @block3_1 = Block.new(256, 3)
      @block3_2 = Block.new(256, 3)
      @block3_3 = Block.new(256, 3)
      @block4_1 = Block.new(512, 3)
      @block4_2 = Block.new(512, 3)
      @block4_3 = Block.new(512, 3)
      @block5_1 = Block.new(512, 3)
      @block5_2 = Block.new(512, 3)
      @block5_3 = Block.new(512, 3)
      @fc1 = Chainer::Links::Connection::Linear.new(nil, out_size: 512, nobias: true)
      @bn_fc1 = Chainer::Links::Normalization::BatchNormalization.new(512)
      @fc2 = Chainer::Links::Connection::Linear.new(nil, out_size: n_classes, nobias: true)
    end
  end

  def call(x)
    # 64 channel blocks:
    h = @block1_1.(x)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.3)
    h = @block1_2.(h)
    h = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(h, 2, stride: 2)

    # 128 channel blocks:
    h = @block2_1.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block2_2.(h)
    h = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(h, 2, stride:2)

    # 256 channel blocks:
    h = @block3_1.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block3_2.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block3_3.(h)
    h = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(h, 2, stride: 2)

    # 512 channel blocks:
    h = @block4_1.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block4_2.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block4_3.(h)
    h = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(h, 2, stride: 2)

    # 512 channel blocks:
    h = @block5_1.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block5_2.(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.4)
    h = @block5_3.(h)
    h = Chainer::Functions::Pooling::MaxPooling2D.max_pooling_2d(h, 2, stride: 2)

    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.5)
    h = @fc1.(h)
    h = @bn_fc1.(h)
    h = Chainer::Functions::Activation::Relu.relu(h)
    h = Chainer::Functions::Noise::Dropout.dropout(h, ratio: 0.5)
    @fc2.(h)
  end
end
