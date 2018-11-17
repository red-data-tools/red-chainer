module Chainer
  module Device
    # Creates device
    #
    # @param [Integer or Chainer::AbstractDevice] device_spec Device specifier.
    #     Negative integer indicates CPU. 0 or positive integer indicates GPU.
    #     If a device object is given, itself is returned.
    # @return [Chainer::AbstractDevice] device object
    def create(device_spec)
      return device_spec if device_spec.kind_of?(AbstractDevice)
      if device_spec.kind_of?(Integer)
        return CpuDevice.new if device_spec < 0
        return GpuDevice.new(device_spec)
      end
      raise "Invalid device_spec: #{device_spec}"
    end
    module_function :create

    # Chainges default device
    #
    # @param [Object] device_spec
    # @see Chainer::Device.create
    def change_default(device_spec)
      @default = create(device_spec)
      @default.use
    end
    module_function :change_default

    # Gets default device
    #
    # @return [Chainer::AbstractDevice] the default device.
    def default
      @default ||= CpuDevice.new
    end
    module_function :default

    # TODO(sonots): Add get_device_from_array after Cumo provides an API
    # to return GPU device ID from Cumo::NArray.
  end

  class AbstractDevice
    def xm
      raise NotImplementedError
    end

    def use
    end
  end

  class CpuDevice < AbstractDevice
    def xm
      Numo
    end

    def ==(other)
      return false unless other.is_a?(CpuDevice)
      true
    end
  end
  
  class GpuDevice < AbstractDevice
    attr_reader :id

    # @param [Integer] id GPU Device ID. If not given, CUDA current device id is used.
    def initialize(id = nil)
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
end
