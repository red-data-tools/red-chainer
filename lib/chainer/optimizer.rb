module Chainer
  class Optimizer
    def setup(link)
      @target = link
      @t = 0
      @epoch = 0
      
      @hooks = {}
    end
  end
end
