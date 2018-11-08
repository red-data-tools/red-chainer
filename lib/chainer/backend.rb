module Chainer
  # Gets an appropriate one from +Numo::NArray+ or +Cumo::NArray+ from given arrays.
  #
  # @param [Array<Chainer::Variable> or Array<Numo::NArray> or Array<Cumo::NArray>] args Values to determine whether Numo or Cumo should be used.
  # @return [Class] +Cumo::NArray+ or +Numo::NArray+ is returned based on the types of the arguments.
  def get_array_module(*args)
    arrays = args.map {|v| v.kind_of?(Chainer::Variable) ? v.data : v }
    if CUDA.available?
      return Cumo if arrays.any? {|a| a.kind_of?(Cumo::NArray) }
    end
    return Numo
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

  class Device
    def xm
      raise 'Not implemented'
    end
  end

  class CpuDevice < Device
    def id
      -1
    end

    def xm
      Numo
    end

    def ==(other)
      return false unless other.is_a?(CpuDevice)
      true
    end
  end
  
  class GpuDevice < Device
    attr_reader :id

    def initialize(id)
      Chainer::CUDA.check_available
      if id < 0
        raise 'GPU Device ID must not be negative'
      end
      @id = id
    end

    def xm
      Cumo
    end

    def ==(other)
      return false unless other.is_a?(GpuDevice)
      id == other.id
    end
  end

  # Gets device
  #
  # @param [Object] device_spec Device specifier. Integer or Device object.
  #     Negative integer indicates CPU. 0 or positive integer indicates GPU.
  # @return [Device] device object
  def get_device(device_spec)
    return device_spec if device_spec.kind_of?(Device)
    if device_spec.kind_of?(Integer)
      return CpuDevice.new if device_spec < 0
      return GpuDevice.new(device_spec)
    end
    raise "Invalid device_spec: #{device_spec}"
  end
  module_function :get_device

  # Sets default device
  #
  # @param [Object] device_spec
  # @see Chainer.set_device
  def set_default_device(device_spec)
    @device = Chainer.get_device(device_spec)
    if @device.id > 0
      Chainer::CUDA.check_available
      Cumo::CUDA::Runtime.cudaSetDevice(@device.id)
    end
  end
  module_function :set_default_device

  # Gets default device
  #
  # @return [Device] device object.
  def get_default_device
    @device ||= CpuDevice.new
  end
  module_function :get_default_device

end
