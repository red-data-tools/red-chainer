module Chainer
  class Configuration
    attr_accessor :enable_backprop, :train

    def initialize
      @enable_backprop = true
      @train = true
    end
  end
end

