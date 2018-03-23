module Chainer
  class Initializer
    attr_accessor :dtype

    def initialize(dtype: nil)
      @dtype = dtype
    end

    def call(array)
      raise NotImplementedError
    end
  end
end

