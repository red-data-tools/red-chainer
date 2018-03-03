module Chainer
  # Gets an appropriate one from +Numo::NArray+ or +Cumo::NArray+.
  #
  # This is almost equivalent to +Chainer::get_array_module+. The differences
  # are that this function can be used even if CUDA is not available and that
  # it will return their data arrays' array module for
  # +Chainer::Variable+ arguments.
  #
  # @param [Array<Chainer::Variable> or Array<Numo::NArray> or Array<Cumo::NArray>] args Values to determine whether Numo or Cumo should be used.
  # @return [Numo::NArray] +Cumo::NArray+ or +Numo::NArray+ is returned based on the types of
  #   the arguments.
  # @todo CUDA is not supported, yet.
  #
  def get_array_module(*args)
    return Numo::NArray
  end
  module_function :get_array_module
end
