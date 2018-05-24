module Chainer
  class Link
    attr_accessor :name

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
        self.instance_variables.each do |name|
          set_attr(name, self.instance_variable_get(name))
        end
      ensure
        @within_init_scope = old_flag
      end
    end

    def set_attr(name, value)
      if within_init_scope && value.kind_of?(Chainer::Parameter)
        value.name = name
        @params << name
        @persistent.delete(name)
      end
    end

    def del_attr(name)
      @params.delete(name)
      @persistent.delete(name)
      self.remove_instance_variable(name)
    end

    def cleargrads
      params do |param|
        param.cleargrad
      end
    end

    # Registers an attribute of a given name as a persistent value.
    # This is a convenient method to register an existing attribute as a persistent value.
    # If `name` has been already registered as a parameter,
    # this method removes it from the list of parameter names and re-registers it as a persistent value.
    #
    # @param [string] name Name of the attribute to be registered.
    def register_persistent(name)
      @persistent << name
      @params.delete(name)
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

    def set_attr(name, value)
      if within_init_scope && value.kind_of?(Chainer::Link)
        if self.respond_to?(name)
          raise TypeError, "cannot register a new link #{name}: attribute exists"
        end
        value.name = name
        @children << name
      end
      super
    end

    def del_attr(name)
      @children.delete(name)
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


  # Composable link with list-like interface.
  #
  # This is another example of compositional link. Unlike :class:`Chainer::Chain`,
  # this class can be used like a list of child links.
  # Each child link is indexed by a non-negative integer,
  # and it maintains the current number of registered child links.
  # The :meth:`add_link` method inserts a new link at the end of the list.
  # It is useful to write a chain with arbitrary number of child links,
  # e.g. an arbitrarily deep multi-layer perceptron.
  class ChainList < Link
    attr_reader :children

    def initialize(*links)
      super()
      @children = []

      links.each do |link|
        add_link(link)
      end
    end

    def set_attr(name, value)
      if within_init_scope && value.kind_of?(Chainer::Link)
        raise TypeError, 'cannot register a new link within a "with chainlist.init_scope:" block.'
      end
      super
    end

    def [](index)
      @children[index]
    end

    def each(&block)
      @children.each(&block)
    end

    def size
      @children.size
    end

    def <<(link)
      add_link(link)
    end

    def add_link(link)
      link.name = @children.size.to_s
      @children << link
    end

    def params(include_uninit: true)
      super(include_uninit: include_uninit) do |param|
        yield param
      end

      @children.each do |link|
        link.params(include_uninit: include_uninit) do |param|
          yield param
        end
      end
    end

    def namedparams(include_uninit: true)
      super(include_uninit: include_uninit) do |ret|
        yield ret
      end
      @children.each_with_index do |link, idx|
        prefix = "/#{idx}"
        link.namedparams(include_uninit: include_uninit) do |path, param|
          yield [prefix + path, param]
        end
      end
    end

    def links(skipself: false)
      unless skipself
        yield self
      end

      @children.each do |child|
        child.links do |link|
          yield link
        end
      end
    end

    def namedlinks(skipself: false)
      unless skipself
        yield '/', self
      end

      @children.each_with_index do |child, idx|
        prefix = "/#{idx}"
        yield prefix, child
        child.namedlinks(skipself: true) do |path, link|
          yield [prefix + path, link]
        end
      end
    end

    def children
      @children.each do |child|
        yield child
      end
    end

    def serialize(serializer)
      super
      @children.each_with_index do |child, idx|
        child.serialize(serialize[idx.to_s])
      end
    end
  end
end
