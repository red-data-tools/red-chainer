class Chainer::Testing::Parameterize
  def product(list)

  end

  def self.product_dict(*parameters)
    parameters.inject do |acc, dicts|
      acc.product(dicts).map{|a, b| a.merge(b) }
    end.map do |data|
      [data.inspect, data]
    end.to_h
  end
end
