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

        if Chainer::CUDA.available?(1)
          Chainer::GpuDevice.new(1).use
          assert Cumo::CUDA::Runtime.cudaGetDevice == 1
        end
      ensure
        Cumo::CUDA::Runtime.cudaSetDevice(orig_device_id)
      end
    end
  end if Chainer::CUDA.available?

  class TestGetDevice < Test::Unit::TestCase
    def test_device
      device = Chainer::CpuDevice.new
      assert Chainer.get_device(device) === device
    end

    def test_negative_integer
      assert Chainer.get_device(-1) == Chainer::CpuDevice.new
    end

    def test_non_negative_integer
      assert Chainer.get_device(0) == Chainer::GpuDevice.new(0)
    end if Chainer::CUDA.available?
  end

  def test_set_get_default_device
    orig_default_device = Chainer.get_default_device
    begin
      device = Chainer::CpuDevice.new
      Chainer.set_default_device(device)
      assert Chainer.get_default_device === device
    ensure
      Chainer.set_default_device(orig_default_device)
    end
  end
end
