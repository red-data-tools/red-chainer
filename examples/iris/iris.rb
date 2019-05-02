require 'numo/narray'
require 'chainer'
require "datasets"

class IrisChain < Chainer::Chain
  L = Chainer::Links::Connection::Linear
  F = Chainer::Functions

  def initialize(n_units, n_out)
    super()
    init_scope do
      @l1 = L.new(nil, out_size: n_units)
      @l2 = L.new(nil, out_size: n_out)
    end
  end

  def call(x, y)
    return F::Loss::MeanSquaredError.mean_squared_error(fwd(x), y)
  end

  def fwd(x)
    h1 = F::Activation::Sigmoid.sigmoid(@l1.(x))
    h2 = @l2.(h1)
    return h2
  end
end

device = Chainer::Device.create(Integer(ENV['RED_CHAINER_GPU'] || -1))
Chainer::Device.change_default(device)
xm = device.xm

model = IrisChain.new(6,3)

optimizer = Chainer::Optimizers::Adam.new
optimizer.setup(model)

iris = Datasets::Iris.new
iris_table = iris.to_table
x = iris_table.fetch_values(:sepal_length, :sepal_width, :petal_length, :petal_width).transpose

# target
y_class = iris_table[:label]

# class index array
# ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
class_name = y_class.uniq
# y => [0, 0, 0, 0, ,,, 1, 1, ,,, ,2, 2]
y = y_class.map{|s|
  class_name.index(s)
}

# y_onehot => One-hot [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],,, [0.0, 1.0, 0.0], ,, [0.0, 0.0, 1.0]]
y_onehot = xm::SFloat.eye(class_name.size)[y, false]

puts "Iris Datasets"
puts "No. [sepal_length, sepal_width, petal_length, petal_width] one-hot #=> class"
x.each_with_index{|r, i|
  puts "#{'%3d' % i} : [#{r.join(', ')}] #{y_onehot[i, false].to_a} #=> #{y_class[i]}(#{y[i]})"
}
# [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]     => 50 data
# [7.0, 3.2, 4.7, 1.4, "Iris-versicolor"] => 50 data
# [6.3, 3.3, 6.0, 2.5, "Iris-virginica"]  => 50 data

x = xm::SFloat.cast(x)
y = xm::SFloat.cast(y)
y_onehot = xm::SFloat.cast(y_onehot)

x_train = x[(1..-1).step(2), true]        #=> 75 data (Iris-setosa : 25, Iris-versicolor : 25, Iris-virginica : 25)
y_train = y_onehot[(1..-1).step(2), true] #=> 75 data (Iris-setosa : 25, Iris-versicolor : 25, Iris-virginica : 25)
x_test = x[(0..-1).step(2), true]         #=> 75 data (Iris-setosa : 25, Iris-versicolor : 25, Iris-virginica : 25)
y_test = y[(0..-1).step(2)]               #=> 75 data (Iris-setosa : 25, Iris-versicolor : 25, Iris-virginica : 25)

puts

# Train
print("Training ")

10000.times{|i|
  print(".") if i % 1000 == 0
  x = Chainer::Variable.new(x_train)
  y = Chainer::Variable.new(y_train)
  model.cleargrads()
  loss = model.(x, y)
  loss.backward()
  optimizer.update()
}

puts

# Test
xt = Chainer::Variable.new(x_test)
yt = model.fwd(xt)
n_row, n_col = yt.data.shape

puts "Result : Correct Answer : Answer <= One-Hot"
ok = 0
n_row.times{|i|
  ans = yt.data[i, true].max_index()
  if ans == y_test[i]
    ok += 1
    printf("OK")
  else
    printf("--")
  end
  printf(" : #{y_test[i].to_i} :")

  puts " #{ans.to_i} <= #{yt.data[i, 0..-1].to_a}"
}
puts "Row: #{n_row}, Column: #{n_col}"
puts "Accuracy rate : #{ok}/#{n_row} = #{ok.to_f / n_row}"
