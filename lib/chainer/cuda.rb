begin
  require 'cumo'
  $chainer_cuda_available = true
rescue LoadError => e
  $chainer_cuda_available = false
end

module Chainer
  module CUDA
    # A trick to make Chainer::CUDA::Cumo::NArray always work
    if $chainer_cuda_available
      Cumo = ::Cumo
    else
      module Cumo
        class NArray
        end
      end
    end

    # Returns whether CUDA is available.
    #
    # @return [Boolean]
    def available?
      $chainer_cuda_available
    end
    module_function :available?

    # Checks if CUDA is available.
    #
    # When CUDA is correctly set up, nothing happens.
    # Otherwise it raises ``RuntimeError``.
    def check_available
      raise 'CUDA environment is not correctly set up' unless available?
    end
    module_function :check_available
  end
end
