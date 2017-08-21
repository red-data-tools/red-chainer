module Chainer
  class Initializer
    def initialize(dtype: nil)
      @dtype = dtype
    end

    def call(array)
      raise NotImplementedError
    end
  end
end

