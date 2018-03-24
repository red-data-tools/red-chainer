# frozen_string_literal: true

module Chainer
  module Testing
    def assert_allclose(expect, actual, atol: 1e-5, rtol: 1e-4)
      # Asserts if some corresponding element of x and y differs too much.
      #
      #   This function can handle both CPU and GPU arrays simultaneously.
      #
      #   Args:
      #       expect: Left-hand-side array.
      #       actual: Right-hand-side array.
      #       atol (float): Absolute tolerance.
      #       rtol (float): Relative tolerance.
      #
      expect = Utils::Array.force_array(expect)
      actual = Utils::Array.force_array(actual)

      # If the expected is 0-dim arrary, extend the dimension to the actual.
      if (expect.shape != actual.shape) and (expect.ndim == 0)
         expect = actual.class.new(actual.shape).fill(expect.to_f)
      end

      actual.each_with_index{|actual_val, *i|
        if (expect[*i].to_f - actual_val.to_f).abs > atol + rtol * expect[*i].abs
          raise "assert_allclose Error\n  expect: #{expect.inspect}\n  actual : #{actual.inspect}\n    (#{i})=> #{(expect - actual).abs.max()} > #{atol + rtol * expect[*i].abs}"
        end
      }
    end
    module_function :assert_allclose
  end
end
