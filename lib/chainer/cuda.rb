begin
  require 'cumo'
  $chainer_cuda_available = true
rescue LoadError => e
  $chainer_cuda_available = false
  # A trick to make Cumo::NArray always exists
  module Cumo
    class NArray; end
    class NMath; end
    class Bit; end
  end
end

module Chainer
  module CUDA
    # Returns whether CUDA is available.
    #
    # @param [Integer or nil] id If a non negative integer is given, check availability of GPU ID.
    # @return [Boolean]
    def available?(id = nil)
      return false unless $chainer_cuda_available
      if id
        raise 'id must be non negative' if id < 0
        @device_count ||= Cumo::CUDA::Runtime.cudaGetDeviceCount
        return @device_count > id
      end
      true
    end
    module_function :available?

    # Checks if CUDA is available.
    #
    # @param [Integer or nil] id If a non negative integer is given, check availability of GPU ID.
    # @raise [RuntimeError] if not available
    def check_available(id = nil)
      raise 'CUDA is not available' unless available?(id)
    end
    module_function :check_available

    # Returns whether cuDNN is available and enabled.
    #
    # To disable cuDNN feature, set an environment variable `RED_CHAINER_CUDNN` to 0.
    #
    # @return [Boolean]
    def cudnn_enabled?
      return @cudnn_enabled unless @cudnn_enabled.nil?
      f = -> () do
        return false unless $chainer_cuda_available
        return false if Integer(ENV.fetch('RED_CHAINER_CUDNN', '1')) == 0
        Cumo::CUDA.const_defined?(:CUDNN) && Cumo::CUDA::CUDNN.available?
      end
      @cudnn_enabled = f.call
    end
    module_function :cudnn_enabled?
  end
end
