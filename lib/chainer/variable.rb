require 'numo/narray'

module Chainer
  class Variable
    attr_accessor :data, :name, :grad, :requires_grad, :node

    def initialize(data:, name: nil, grad: nil, requires_grad: true)
      validate(data: data) 
      self.data = [data]
      self.name = name
      self.grad = grad
      self.requires_grad = requires_grad
    end

    def data
      return @data[0]
    end

    private

    def validate(data:)
      unless data.is_a?(Numo::NArray)
        raise TypeError, "Numo::NArray are expected."
      end
    end
  end
end
