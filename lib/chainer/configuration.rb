module Chainer
  class Configuration
    attr_accessor :enable_backprop

    def initialize
      @enable_backprop = true
    end
  end
end

