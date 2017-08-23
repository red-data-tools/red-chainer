module Chainer
  class Hyperparameter
    attr_reader :parent

    def initialize(parent: nil)
      @parent = parent
    end

    def method_missing(name)
      @parent.send(name)
    end

    def get_dict
      d = @parent.nil? ? {} : @parent.get_dict
      self.instance_variables.each do |m|
        unless m == :@parent
          d[m.to_s.delete('@')] = self.instance_variable_get(m)
        end
      end
      d
    end
  end
end
