# Red Chainer : A deep learning framework

A flexible framework for neural network for Ruby

## Description

It ported python's [Chainer](https://github.com/chainer/chainer) with Ruby.

## Requirements

* Ruby 2.4 or later

## Installation

Add this line to your application's Gemfile:

```bash
gem 'red-chainer'
```

And then execute:

```bash
$ bundle
```

Or install it yourself as:

```bash
$ gem install red-chainer
```

## Usage

### Run MNIST example

MNIST sample program is [here](./examples/mnist/mnist.rb)

```bash
# when install Gemfile
$ bundle exec ruby examples/mnist/mnist.rb
# when install yourself
$ ruby examples/mnist/mnist.rb
```

### Run MNIST example with GPU

On GPU machine, add `gem 'cumo'` on Gemfile and do `bundle install`.

Run the example with `--gpu` option whose value indicates GPU device ID such as:

```
$ bundle exec ruby examples/mnist/mnist.rb --gpu 0
```

## Development

### Run tests

```
$ bundle exec ruby test/run_test.rb
```

### Run tests with Cumo

On GPU machine, add `gem 'cumo'` on Gemfile and do `bundle install`.

Run tests with `RED_CHAINER_GPU` environment variable whose value indicates GPU device ID such as:

```
$ bundle exec env RED_CHAINER_GPU=0 ruby test/run_test.rb
```

## License

The MIT license. See [LICENSE.txt](./LICENSE.txt) for details.

## Red Chainer implementation status

|    |  Chainer 2.0<br>(Initial ported version)  | Red Chainer (0.3.1) | example |
| ---- | ---- | ---- | ---- |
|  [activation](https://github.com/red-data-tools/red-chainer/tree/master/lib/chainer/functions/activation)  |  15  | 5 | LogSoftmax, ReLU, LeakyReLU, Sigmoid, Tanh |
|  [loss](https://github.com/red-data-tools/red-chainer/tree/master/lib/chainer/functions/loss)  |  17  | 2 | SoftMax, MeanSquaredError |
|  [optimizer](https://github.com/red-data-tools/red-chainer/tree/master/lib/chainer/optimizers)  |  9  | 2 | Adam, MomentumSGDRule |
|  [connection](https://github.com/red-data-tools/red-chainer/tree/master/lib/chainer/functions/connection)  |  12  | 2 | Linear, Convolution2D |
|  [pooling](https://github.com/red-data-tools/red-chainer/tree/master/lib/chainer/functions/pooling)  |  14  | 3 | Pooling2D, MaxPooling2D, AveragePooling2D |
|  [example](https://github.com/red-data-tools/red-chainer/tree/master/examples)  |  31  | 3 | MNIST, Iris, CIFAR |
|  GPU  | use CuPy  | use [Cumo](https://github.com/sonots/cumo) ||
