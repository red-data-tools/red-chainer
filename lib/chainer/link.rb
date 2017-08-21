module Chainer
  class Link
    def initialize
      @params = []
      @persistent = []
      @within_init_scope = false
      @name = nil
    end

    def within_init_scope
      @within_init_scope || false
    end

    def init_scope
      old_flag = self.within_init_scope
      @within_init_scope = true

      begin
        yield
      ensure
        @within_init_scope = old_flag
      end
    end
  end

  class Chain < Link
    def initialize
      super
      @children = []
    end
  end
end
