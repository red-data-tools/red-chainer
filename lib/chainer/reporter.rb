module Chainer
  module ReportService
    @@reporters = []
  end

  class Reporter
    include ReportService

    attr_accessor :observer_names, :observation

    def initialize
      @observer_names = {}
      @observation = {}
    end

    def self.save_report(values, observer=nil)
      reporter = @@reporters[-1]
      reporter.report(values, observer)
    end

    def self.report_scope(observation)
      current = @@reporters[-1]
      old = current.observation
      current.observation = observation
      yield
      current.observation = old
    end

    def report(values, observer=nil)
      # TODO: keep_graph_on_report option
      if observer
        observer_id = observer.object_id
        unless @observer_names.keys.include?(observer_id)
          raise "Given observer is not registered to the reporter."
        end
        observer_name = @observer_names[observer_id]
        values.each do |key, value|
          name = "#{observer_name}/#{key}"
          @observation[name] = value
        end
      else
        @observation.update(values)
      end
    end

    def add_observer(name, observer)
      @observer_names[observer.object_id] = name
    end

    def add_observers(prefix, observers, skipself: true)
      observers.call(skipself: skipself) do |name, observer|
        @observer_names[observer.object_id] = "#{prefix}#{name}"
      end
    end

    def scope(observation)
      @@reporters << self
      old = @observation
      @observation = observation
      yield
      @observation = old
      @@reporters.pop
    end
  end

  class Summary
    def initialize
      @x = 0
      @x2 = 0
      @n = 0
    end

    # Adds a scalar value.
    # Args:
    #   value: Scalar value to accumulate.
    def add(value)
      @x += value
      @x2 += value * value
      @n += 1
    end

    # Computes the mean.
    def compute_mean
      @x.to_f / @n
    end

    # Computes and returns the mean and standard deviation values.
    # Returns:
    #   array: Mean and standard deviation values.    
    def make_statistics
      mean = @x / @n
      var = @x2 / @n - mean * mean
      std = Math.sqrt(var)
      [mean, std]
    end
  end

  # Online summarization of a sequence of dictionaries.
  # ``DictSummary`` computes the statistics of a given set of scalars online.
  # It only computes the statistics for scalar values and variables of scalar values in the dictionaries.
  class DictSummary
    def initialize
      @summaries = Hash.new { |h,k| h[k] = Summary.new }
    end

    # Adds a dictionary of scalars.
    # Args:
    #   d (dict): Dictionary of scalars to accumulate. Only elements of
    #             scalars, zero-dimensional arrays, and variables of
    #             zero-dimensional arrays are accumulated.
    def add(d)
      d.each do |k, v|
        v = v.data if v.kind_of?(Chainer::Variable)
        if v.class.method_defined?(:to_i) || (v.class.method_defined?(:ndim) && v.ndim == 0)
          @summaries[k].add(v)
        end 
      end
    end

    # Creates a dictionary of mean values.
    # It returns a single dictionary that holds a mean value for each entry added to the summary.
    # 
    # Returns:
    #   dict: Dictionary of mean values.
    def compute_mean
      @summaries.each_with_object({}) { |(name, summary), h| h[name] = summary.compute_mean }
    end

    # Creates a dictionary of statistics.
    # It returns a single dictionary that holds mean and standard deviation
    # values for every entry added to the summary. For an entry of name
    # ``'key'``, these values are added to the dictionary by names ``'key'`` and ``'key.std'``, respectively.
    # 
    # Returns:
    #   dict: Dictionary of statistics of all entries.
    def make_statistics
      stats = {}
      @summaries.each do |name, summary|
        mean, std = summary.make_statistics
        stats[name] = mean
        stats[name + '.std'] = std
      end
      stats
    end
  end
end
