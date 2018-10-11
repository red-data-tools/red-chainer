module Chainer
  # Gets an appropriate one from +Numo::NArray+ or +Cumo::NArray+ from given arrays.
  #
  # @param [Array<Chainer::Variable> or Array<Numo::NArray> or Array<Cumo::NArray>] args Values to determine whether Numo or Cumo should be used.
  # @return [Class] +Cumo::NArray+ or +Numo::NArray+ is returned based on the types of the arguments.
  def get_array_module(*args)
    arrays = args.map {|v| v.kind_of?(Chainer::Variable) ? v.data : v }
    if CUDA.available?
      return Cumo::NArray arrays.any? {|a| a.kind_of?(Cumo::NArray) }
    end
    return Numo::NArray
  end
  module_function :get_array_module

  # Returns true if the argument is either of +Numo::NArray+ or +Cumo::NArray+.
  #
  # @param [Object] obj
  # @return [Boolean]
  def array?(obj)
    if CUDA.available?
      return true if obj.kind_of?(Cumo::NArray)
    end
    return true if obj.kind_of?(Numo::NArray)
    false
  end
  module_function :array?
end
