# frozen_string_literal: true

require 'chainer'

class TestDevice < Test::Unit::TestCase
  class TestCpuDevice < Test::Unit::TestCase
    def test_xm
      assert Chainer::CpuDevice.new.xm == Numo
    end

    def test_eq
      assert Chainer::CpuDevice.new == Chainer::CpuDevice.new
    end

    def test_use
      Chainer::CpuDevice.new.use # nothing raised
    end
  end

  class TestGpuDevice < Test::Unit::TestCase
    def setup
      require_gpu
    end

    def test_xm
      assert Chainer::GpuDevice.new.xm == Cumo
    end

    def test_id
      assert Chainer::GpuDevice.new(0).id == 0
    end

    def test_eq
      assert Chainer::GpuDevice.new == Chainer::GpuDevice.new(0)
    end

    def test_use
      orig_device_id = Cumo::CUDA::Runtime.cudaGetDevice
      begin
        Chainer::GpuDevice.new(0).use
        assert Cumo::CUDA::Runtime.cudaGetDevice == 0
      ensure
        Cumo::CUDA::Runtime.cudaSetDevice(orig_device_id)
      end
    end

    def test_use_1
      require_gpu(1)
      orig_device_id = Cumo::CUDA::Runtime.cudaGetDevice
      begin
        Chainer::GpuDevice.new(1).use
        assert Cumo::CUDA::Runtime.cudaGetDevice == 1
      ensure
        Cumo::CUDA::Runtime.cudaSetDevice(orig_device_id)
      end
    end
  end

  class TestCreateDevice < Test::Unit::TestCase
    def test_device
      device = Chainer::CpuDevice.new
      assert Chainer::Device.create(device) === device
    end

    def test_negative_integer
      assert Chainer::Device.create(-1) == Chainer::CpuDevice.new
    end

    def test_non_negative_integer
      require_gpu
      assert Chainer::Device.create(0) == Chainer::GpuDevice.new(0)
    end
  end

  def test_change_get_default
    orig_default = Chainer::Device.default
    begin
      device = Chainer::CpuDevice.new
      Chainer::Device.change_default(device)
      # Chainer::Device.set_default(device)
      # Chainer::Device.default = device
      assert Chainer::Device.default === device
    ensure
      Chainer::Device.change_default(orig_default)
    end
  end
end
