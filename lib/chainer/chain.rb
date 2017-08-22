module Chainer
  class Chain << Link
    def initialize
      super
      @children = []
    end
  end
end
