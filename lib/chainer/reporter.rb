module Chainer
  class Reporter
    def initialize
      @observer_names = {}
      @observation = {}
    end

    def add_observer(name, observer)
      @observer_names[observer.object_id] = name
    end

    def add_observers(prefix, observers)
      observers.each do |(name, observer)|
        @observer_names[observer.object_id] = prefix + name
      end
    end

    def scope(observation)
      old = @observation
      @observation = observation
      yield
      @observation = old
    end
  end

  class DictSummary
    def initialize
      @summaries = Hash.new { |h,k| h[k] = [] }
    end
  end
end
