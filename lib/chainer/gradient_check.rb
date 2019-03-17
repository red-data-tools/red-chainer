module Chainer
  def _copy_arrays(xs)
    xs.map{|x| Chainer.array?(x) ? x.dup : x}
  end

  # Computes numerical gradient by finite differences.
  #
  # This function is used to implement gradient check. For usage example, see
  # unit tests of Chainer::Functions.
  #
  # @param [function] f Ruby function with no arguments that runs forward
  #   computation and returns the result.
  # @param [Array<Arrays>] inputs Array of arrays that should be treated as
  #   inputs. Each element of them is slightly modified to realize numerical
  #   gradient by finite differences.
  # @param [Array<Arrays>] grad_outputs Array of arrays that are treated as
  #   output gradients.
  # @param [Float] eps Epsilon value of finite differences.
  # @return [Array] Numerical gradient arrays corresponding to +inputs+.
  #
  def numerical_grad(f, inputs, grad_outputs, eps=0.001)
    raise unless eps > 0
    inputs = inputs.to_a
    grad_outputs = grad_outputs.to_a
    grads = inputs.map{|x| x.new_zeros()}

    if inputs[0].ndim < 2
      tmp = [[inputs[0], grads[0]]]
    else
      tmp = (0...inputs[0].shape[0]).map{|i|[inputs[0][i, false], grads[0][i, false]]}
    end

    tmp.each do |x, gx|
      x.each_with_index{|xx, *i|
        orig = x[*i].to_f   # hold original value
        x[*i] = orig + eps
        ys1 = _copy_arrays(f.call)
        x[*i] = orig - eps
        ys2 = _copy_arrays(f.call)
        x[*i] = orig

        ys1.zip(ys2, grad_outputs).each do |y1, y2, gy|
          if !gy.nil?
            # TODO: Subtracting between empty matrices loses shape
            # For example:
            #   x = Numo::DFloat.new(0, 5)
            #   x.shape
            #   # => [0, 5]
            #   (x - x).shape
            #   # => [0, 0]
            if (Chainer.array?(y1) && y1.empty?) || (Chainer.array?(y2) && y2.empty?) || (Chainer.array?(gy) && gy.empty?)
              dot = 0
            elsif Chainer.array?((y1 - y2) * gy)
              dot = ((y1 - y2) * gy).sum()
            else
              dot = ((y1 - y2) * gy).inject(:+)
            end
            gx[*i] += dot / (2*eps).to_f
          end
        end
      }
    end

    return grads
  end

  def _as_tuple(x)
    if x.is_a? Array
      return x
    else
      return [x]
    end
  end

  # Test backward procedure of a given function.
  #
  # This function automatically check backward-process of given function.
  # For example, when you have a +Chainer::Function+ class +MyFunc+,
  # that gets two arguments and returns one value, you can make its test like this:
  #
  #   def test_my_func(self):
  #     func = MyFunc()
  #     x1_data = Numo::NArray[...]
  #     x2_data = Numo::NArray[...]
  #     gy_data = Numo::NArray[...]
  #     check_backward(func, [x1_data, x2_data], gy_data)
  #
  # This method creates +Chainer::Variable+ objects with +x_data+
  # and calls +func+ with the +Chainer::Variable+ s to get its result
  # as +Chainer::Variable+.
  # Then, it sets +y_grad+ array to +grad+ attribute of the result and
  # calls +backward+ method to get gradients of the inputs.
  # To check correctness of the gradients, the function calls
  # +numerical_grad+ to calculate numerically the gradients and compares
  # the types of gradients with +Chainer::Testing.assert_allclose+.
  # If input objects (+x1_data+ or/and +x2_data+ in this example) represent
  # integer variables, their gradients are ignored.
  #
  # You can simplify a test when +MyFunc+ gets only one argument:
  #
  #   check_backward(func, x1_data, gy_data)
  #
  # If +MyFunc+ is a loss function which returns a zero-dimensional
  # array, pass +nil+ to +gy_data+. In this case, it sets +1+ to
  # +grad+ attribute of the result:
  #
  #   check_backward(my_loss_func, [x1_data, x2_data], nil)
  #
  # If +MyFunc+ returns multiple outputs, pass all gradients for outputs as a Array:
  #
  #   gy1_data = Numo::NArray[...]
  #   gy2_data = Numo::NArray[...]
  #   check_backward(func, x1_data, [gy1_data, gy2_data])
  #
  # You can also test a +Chainer::Link+.
  # To check gradients of parameters of the link, set a Array of the parameters
  # to +params+ arguments:
  #
  #   check_backward(my_link, [x1_data, x2_data], gy_data, [my_link.W, my_link.b])
  #
  # Note that +params+ are not +Numo::NArray+ s,
  # but +Chainer::Variables+ s.
  #
  # Function objects are acceptable as +func+ argument:
  #
  #   check_backward(lambda{|x1, x1| f(x1, x2)}, [x1_data, x2_data], gy_data)
  #
  # @note
  #  +func+ is called many times to get numerical gradients for all inputs.
  #  This function doesn't work correctly when +func+ behaves randomly as
  #  it gets different gradients.
  # @param [Method, Proc] func A function which gets +Chainer::Variable+ s
  #   and returns +Chainer::Variable+ s. +func+ must returns
  #   a Array of +Chainer::Variable+ s or one
  #   +Chainer::Variable+. You can use +Chainer::Function+
  #   object, +Chainer::Link+ object or a function satisfying the
  #   condition.
  # @param [Numo::NArray or Array<Numo::NArray>] x_data A set of +Numo::NArray+ s to be
  #   passed to +func+. If +x_data+ is one +Numo::NArray+ object, it is
  #   treated as +(x_data,)+.
  # @param [Numo::NArray or Array<Numo::NArray> or nil] y_grad A set of +Numo::NArray+ s representing gradients of return-values of
  #   +func+. If +y_grad+ is one +Numo::NArray+ object, it is
  #   treated as +(y_grad,)+. If +func+ is a loss-function,
  #   +y_grad+ should be set to +nil+.
  # @param [Chainer::Variable or Array<Chainder::Variable>] params  A set of +Chainer::Variable+ s whose gradients are checked.
  #   When +func+ is a +Chainer::Link+ object,
  #   set its parameters as +params+.
  #   If +params+ is one +Chainer::Variable+ object,
  #   it is treated as +(params,)+.
  # @param [Float] eps Epsilon value to be passed to +numerical_grad+.
  # @param [Float] atol Absolute tolerance to be passed to +Chainer::Testing.assert_allclose+.
  # @param [Float] rtol Relative tolerance to be passed to +Chainer::Testing.assert_allclose+.
  # @param [Array<Boolean>] no_grads Flag to skip variable for gradient assertion.
  #   It should be same length as +x_data+.
  # @param [Numo::NArray.class] dtype +x_data+ and +y_grad+ are casted to this
  #   dtype when calculating numerical gradients. Only float types and
  #   +nil+ are allowed.
  # @see
  #   .numerical_grad
  #
  def check_backward(func, x_data, y_grad, params=[], eps: 0.001, atol: 1e-5, rtol: 1e-4, no_grads: nil, dtype: nil)
    x_data = _as_tuple(x_data)
    xm = Chainer.get_array_module(*x_data)
    if !y_grad.nil?
      y_grad = _as_tuple(y_grad)
    end

    params = _as_tuple(params)
    xs = x_data.map{|x| Chainer::Variable.new(x)}
    y = func.(*xs)
    y = _as_tuple(y)
    y = Chainer::Functions::Math::Identity.new.apply(y)

    if !y_grad.nil?
      if (y).size != (y_grad).size
        raise TypeError, "`y_grad` must have the same length of output values"
      end

      y.zip(y_grad).each do |iy, igy|
        if igy.is_a?(Chainer::Variable)
          iy.grad_var = igy
        else
          iy.grad = igy
        end
      end
    else
      if (y).size != 1
        raise TypeError, "When `y_grad` is `nil`, the function must return azero-dimentional array"
      end
      y_grad = [1]
    end

    # We only need to call `backward` for one result `Chainer::Variable`.
    # `Chainer::Variable.backward` method calls `Chainer::Function.backward` of its creator.
    y[0].backward()

    param_data = params.map { |p| p.data }
    if dtype.nil?
      casted_xs = x_data.map { |x| Chainer::Variable.new(x) }
    else
      raise '`dtype` is allowed only float type' if dtype != xm::DFloat && dtype != xm::SFloat
      casted_xs = x_data.map { |x| x.is_a?(Numo::NArray) ? Chainer::Variable.new(x.cast_to(dtype)) : x  }
    end

    if no_grads.nil?
      no_grads = xs.map { |x| x.dtype != Numo::SFloat && x.dtype != Numo::DFloat }
    else
      raise "Length of no_grads param and xs should be same." if no_grads.size != xs.size
    end

    casted_data = casted_xs.map { |x| x.data.dup }

    no_grads.zip(xs).each do |skip, x|
      if skip
        raise "x.grad is not nil" if  x.grad != nil
      else
        raise 'gradients of some arguments are not calculated' if x.grad.nil?
      end
    end

    if dtype.nil?
      one = Numo::DFloat.new().fill(1.0)
    else
      one = dtype.new().fill(1.0)
    end

    g = lambda do
      # This functions is called twice in `numerical_grad`.
      # `one` is `1 + epsilon` or `1 - epsilon` in these calls.
      # See the document of `numerical_grad`.
      no_grads.zip(casted_xs, casted_data).each do |skip, cx, data|
        next if skip || cx.data.empty?
        # astype is require to store data with the given type
        data = (one * data).cast_to(data.class)
        cx.data = data
      end

      params.zip(param_data).each do |param, data|
        if !dtype.nil?
          param_dtype = dtype
        else
          param_dtype = param.dtype
        end
        # The inner astype is required to calculates __mul__ in
        # `param_type` when data is low accuracy float.
        # The outer one is require to store data with the given type.
        param.data = (one * data.cast_to(param_dtype)).cast_to(param_dtype)
      end

      ys = func.(*casted_xs)
      ys = _as_tuple(ys)
      ys_data = ys.map { |y| y.data }
      no_grads.zip(casted_xs, casted_data).each do |skip, cx, data|
        next if skip
        cx.data = data
      end
      params.zip(param_data).each do |param, data|
        param.data = data
      end
      ys_data
    end

    gx, = numerical_grad(g, [one], y_grad, eps)
    gx_accum = 0

    no_grads.zip(xs, casted_xs).each do |skip, x, cx|
      next if skip
      gxi = x.grad.flatten.dup
      cxi = cx.data.flatten.dup
      unless dtype.nil?
        gxi = gxi.cast_to(dtype)
        cxi = cxi.cast_to(dtype)
      end
      gx_accum += gxi.empty? ? 0 : gxi.dot(cxi)
    end

    params.each do |p|
      gpi = p.grad.flatten.dup
      pi = p.data.flatten.dup
      unless dtype.nil?
        gpi = gpi.cast_to(dtype)
        pi = pi.cast_to(dtype)
      end
      gx_accum += gpi.dot(pi)
    end

    Chainer::Testing.assert_allclose(gx, gx_accum, atol: atol, rtol: rtol)
  end

  def check_double_backward(func, x_data, y_grad, x_grad_grad, params=[], params_grad_grad=[], eps: 1e-3, atol: 1e-4, rtol: 1e-3, no_grads: nil, dtype: nil)
    x_data = _as_tuple(x_data)
    n_x = x_data.size

    first_order_grad = -> *inputs do
      xs = inputs[0...n_x]
      gys = inputs[n_x..-1]

      y = _as_tuple(func.(*xs))
      # Let all elements of y share the same creator.
      # See the comment in check_backward.
      y = Chainer::Functions::Math::Identity.new.apply(y)
      if !gys.nil?
        if (y).size != (gys).size
          raise TypeError, "`gys` must have the same length of output values"
        end

        y.zip(gys).each do |iy, igy|
          if igy.is_a?(Chainer::Variable)
            iy.grad_var = igy
          else
            iy.grad = igy
          end
        end
      else
        if (y).size != 1
          raise TypeError, "When `gys` is `nil`, the function must return azero-dimentional array"
        end
        gys = [1]
      end
      y[0].backward

      ret = xs.map { |x| x.grad_var }
      xs.each { |x| x.grad_var = nil }
      ret
    end

    inputs = x_data + _as_tuple(y_grad)
    check_backward(first_order_grad, inputs, x_grad_grad, params=params, eps: eps, atol: atol, rtol: rtol, no_grads: no_grads, dtype: dtype)
  end

  module_function :_copy_arrays, :numerical_grad, :_as_tuple, :check_backward, :check_double_backward
end
