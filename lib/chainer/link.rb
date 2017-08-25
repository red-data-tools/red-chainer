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

    def cleargrads
      params.each do |param|
        param.cleargrad
      end
    end

    def params(include_uninit: true)
      @params.each do |name|
        data = self.send(name).data
        if include_uninit || data
          yield self.send(name)
        end
      end
    end

    def namedparams(include_uninit: true)
      @params.each do |name|
        if include_uninit || self.send(name).data
          yield ['/' + name, self.send(name)]
        end
      end
    end
  end

  class Chain < Link
    def initialize
      super
      @children = []
    end

    def params(include_uninit: true)
      super(include_uninit: include_uninit) do |param|
        yield param
      end
      
      @children.each do |name|
        self.send(name).params(include_uninit: include_uninit) do |param|
          yield param
        end
      end
    end

    def namedparams(include_uninit: true)
      super(include_uninit: include_uninit) do |param|
        yield ret
      end

      @children.each do |name|
        prefix = "/#{name}"
        self.send(name).namedparams(include_uninit: include_uninit).each do |(path, param)|
          yield [prefix + path, param]
        end
      end
    end
  end
end
