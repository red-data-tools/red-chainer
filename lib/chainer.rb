require "weakref"

require "chainer/version"

require 'chainer/configuration'
require 'chainer/function'
require 'chainer/variable'
require 'chainer/variable_node'
require 'chainer/utils/argument'
require 'chainer/utils/variable'
require 'chainer/utils/array'
require 'chainer/functions/math/basic_math'

require 'numo/narray'

module Chainer
  def self.configure
    yield(configuration)
  end

  def self.configuration
    @configuration ||= Configuration.new
  end
end

