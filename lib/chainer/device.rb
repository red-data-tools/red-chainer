module Chainer
  class Device
    def xm
      raise NotImplementedError
    end

    def use
    end
  end

  class CpuDevice < Device
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

    # @param [Integer] id GPU Device ID. If not given, CUDA current device id is used.
    def initialize(id: nil)
      Chainer::CUDA.check_available
      id ||= Cumo::CUDA::Runtime.cudaGetDevice
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

    # Sets CUDA current device with owned GPU Device ID
    def use
      Cumo::CUDA::Runtime.cudaSetDevice(@id)
    end
  end

  # Gets device
  #
  # @param [Integer or Chainer::Device] device_spec Device specifier.
  #     Negative integer indicates CPU. 0 or positive integer indicates GPU.
  #     If a device object is given, itself is returned.
  # @return [Chainer::Device] device object
  def get_device(device_spec)
    return device_spec if device_spec.kind_of?(Device)
    if device_spec.kind_of?(Integer)
      return CpuDevice.new if device_spec < 0
      return GpuDevice.new(id: device_spec)
    end
    raise "Invalid device_spec: #{device_spec}"
  end
  module_function :get_device

  # Sets default device
  #
  # @param [Object] device_spec
  # @see Chainer.set_device
  def set_default_device(device_spec)
    @default_device = Chainer.get_device(device_spec)
    @default_device.use
  end
  module_function :set_default_device

  # Gets default device
  #
  # @return [Chainer::Device] the default device.
  def get_default_device
    @default_device ||= CpuDevice.new
  end
  module_function :get_default_device

  # TODO(sonots): Add get_device_from_array after Cumo provides an API
  # to return GPU device ID from Cumo::NArray.
end
