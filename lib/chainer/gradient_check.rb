module Chainer
  def _copy_arrays(xs)
    xp = Chainer::get_array_module(*xs)
    xs.map{|x| (x.is_a? Numo::NArray) ? x.dup : x}
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
    xp = Numo::NArray
    grads = inputs.map{|x| x.new_zeros()}

    if inputs[0].ndim < 2
      tmp = [[inputs[0], grads[0]]]
    else
      tmp = (0...inputs[0].shape[0]).map{|i|[inputs[0][i, false], grads[0][i, false]]}
    end

    tmp.each do |x, gx|
      x.each_with_index{|xx, *i|
        orig = x[*i]   # hold original value
        x[*i] = orig + eps
        ys1 = _copy_arrays(f.call(x))
        x[*i] = orig - eps
        ys2 = _copy_arrays(f.call(x))
        x[*i] = orig

        ys1.zip(ys2, grad_outputs).each do |y1, y2, gy|
          if !gy.nil?
            if  ((y1 - y2) * gy).is_a? Numo::NArray
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
    if !y_grad.nil?
      y_grad = _as_tuple(y_grad)
    end

    params = _as_tuple(params)
    xs = x_data.map{|x| Chainer::Variable.new(x)}
    y = func.(*xs)
    y = _as_tuple(y)
    y = Chainer::Functions::Math::Identity.identity.apply(y)

    if !y_grad.nil?
      if (y).size != (y_grad).size
        raise TypeError, "`y_grad` must have the same length of output values"
      end

      y.zip(y_grad).each do |iy, igy|
        iy.grad = igy
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

    if dtype.nil?
      casted_xs = x_data.map{|x| Chainer::Variable.new(x)}
    else
      if (dtype != Numo::DFloat) and (dtype != Numo::SFloat)
        raise TypeError, "`dtype` is allowed only float type"
      end
      if (params).size > 0
        raise TypeError, "`dtype` is available only if `params` is empty"
      end
      casted_xs = x_data.map{|x|
                    if x.class == Numo::DFloat or x.class == Numo::SFloat
                      Chainer::Variable.new(dtype.cast(x))
                    else
                      Chainer::Variable.new(x)
                    end
                  }
    end

    f = lambda do |_|
      ys = func.(*casted_xs)
      ys = _as_tuple(ys)
      return ys.map{|y| y.data}.to_a
    end

    if no_grads.nil?
      no_grads = xs.map{|x| (x.dtype != Numo::DFloat) and (x.dtype != Numo::SFloat)}
    else
      if no_grads.size != xs.size
        raise TypeError, "Length of no_grads param and xs should be same."
      end
    end

    no_grads.zip(xs, casted_xs).each do |skip, x, cx|
      if skip
        raise unless x.grad.nil?
        next
      end
      gx, = numerical_grad(f, [cx.data], y_grad, eps)
      Chainer::Testing.assert_allclose(x.grad, gx, atol: atol, rtol: rtol)
      if dtype.nil?
        raise unless gx.class == x.grad.class
      else
        if ((gx.class != Numo::DFloat) and (gx.class != Numo::SFloat)) and (gx.class != dtype)
           raise
        end
      end
    end

    params.each do |p|
      gp, = numerical_grad(f, [p.data], y_grad, eps)
      Chainer::Testing.assert_allclose(p.grad, gp, atol: atol, rtol: rtol)
      raise unless gp.dtype === p.grad.dtype
    end
  end
  module_function :_copy_arrays, :numerical_grad, :_as_tuple, :check_backward
end
