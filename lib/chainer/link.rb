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
        set_attr
      ensure
        @within_init_scope = old_flag
      end
    end

    def set_attr
      self.instance_variables.each do |name|
        value = self.instance_variable_get(name)
        if value.instance_of?(Chainer::Parameter)
          @params << name
          @persistent.delete(name)
        end
      end
    end

    def cleargrads
      params do |param|
        param.cleargrad
      end
    end

    def params(include_uninit: true)
      @params.map do |name|
        data = self.instance_variable_get(name).data
        if include_uninit || data
          yield self.instance_variable_get(name)
        end
      end
    end

    def namedparams(include_uninit: true)
      @params.each do |name|
        if include_uninit || self.instance_variable_get(name).data
          yield ['/' + name.to_s, self.instance_variable_get(name)]
        end
      end
    end

    def namedlinks(skipself: false)
      yield('/', self) unless skipself
    end

    def serialize(serializer)
      d = self.instance_variables.each_with_object({}) { |sym, h| h[sym] = self.instance_variable_get(sym) }
      @params.each do |name|
        param = d[name]
        data = serializer.(name.to_s, param.data)
        if param.data.nil? && !data.nil?
          # Initialize the parameter here
          param.init(data.shape)
          if param.data.is_a?(Numo::NArray)
            param.data.store(data)
          else
            param.data.set(Numo::NArray.cast(data))
          end
        end
      end

      @persistent.each do |name|
        d[name] = serializer.(name.to_s, d[name])
      end
    end
  end

  class Chain < Link
    def initialize
      super
      @children = []
    end

    def set_attr
      self.instance_variables.each do |name|
        value = self.instance_variable_get(name)
        if value.kind_of?(Chainer::Link)
          @children << name
        end
      end
      super
    end

    def params(include_uninit: true)
      super(include_uninit: include_uninit) do |param|
        yield param
      end

      @children.each do |name|
        self.instance_variable_get(name).params(include_uninit: include_uninit) do |param|
          yield param
        end
      end
    end

    def namedparams(include_uninit: true)
      super(include_uninit: include_uninit) do |param|
        yield param
      end

      @children.each do |name|
        prefix = "/#{name}"
        self.instance_variable_get(name).namedparams(include_uninit: include_uninit) do |(path, param)|
          yield [prefix + path, param]
        end
      end
    end

    def namedlinks(skipself: false)
      yield('/' , self) unless skipself
      d = self.instance_variables.each_with_object({}) { |sym, h| h[sym] = self.instance_variable_get(sym) }
      @children.each do |name|
        child = d[name.to_sym]
        prefix = '/' + name.to_s
        yield(prefix, child)
        d[name].namedlinks(skipself: true) do |path, link|
          yield(prefix + path, link)
        end
      end
    end

    def serialize(serializer)
      super(serializer)
      d = self.instance_variables.each_with_object({}) { |sym, h| h[sym] = self.instance_variable_get(sym) }
      @children.each do |name|
        d[name].serialize(serializer[name.to_s])
      end
    end
  end
end
