module Chainer::Links::Connection
  class LSTMBase < Chainer::Chain
    attr_reader :upward, :lateral

    def initialize(in_size, out_size: nil, lateral_init: nil,
                 upward_init: nil, bias_init: 0, forget_bias_init: 1)

      super()

      unless out_size
        out_size = in_size
        in_size = nil
      end

      @state_size = out_size
      @lateral_init = lateral_init
      @upward_init = upward_init
      @bias_init = bias_init
      @forget_bias_init = forget_bias_init

      init_scope do
        @upward = Chainer::Links::Connection::Linear.new(in_size, out_size: 4 * out_size, initial_w: 0)
        @lateral = Chainer::Links::Connection::Linear.new(out_size, out_size: 4 * out_size, initial_w: 0, nobias: true)
        initialize_params if in_size
      end
    end

    def initialize_params
      lateral_init = Chainer::Initializers.get_initializer(@lateral_init)
      upward_init = Chainer::Initializers.get_initializer(@upward_init)

      (0...(4 * @state_size)).step(@state_size).each do |i|
        @lateral.w.data[i...(i + @state_size), true] = lateral_init.(@lateral.w.data[i...(i + @state_size), true])
        @upward.w.data[i...(i + @state_size), true] = upward_init.(@upward.w.data[i...(i + @state_size), true])
      end

      idx = Numo::Int32[*(0...(4*@state_size)).step(4)]
      (a, i, f, o) = Chainer::Functions::Activation::LSTM.extract_gates(@upward.b.data.reshape(1, 4 * @state_size, 1))
      @upward.b.data[idx] = Chainer::Initializers.get_initializer(@bias_init).(a)
      @upward.b.data[idx + 1] = Chainer::Initializers.get_initializer(@bias_init).(i)
      @upward.b.data[idx + 2] = Chainer::Initializers.get_initializer(@forget_bias_init).(f)
      @upward.b.data[idx + 3] = Chainer::Initializers.get_initializer(@bias_init).(o)
    end
  end

  class StatelessLSTM < LSTMBase

    # """Stateless LSTM layer.

    # This is a fully-connected LSTM layer as a chain. Unlike the
    # :func:`~chainer.functions.lstm` function, this chain holds upward and
    # lateral connections as child links. This link doesn't keep cell and
    # hidden states.

    # Args:
    #     in_size (int or None): Dimension of input vectors. If ``None``,
    #         parameter initialization will be deferred until the first forward
    #         data pass at which time the size will be determined.
    #     out_size (int): Dimensionality of output vectors.

    # Attributes:
    #     upward (chainer.links.Linear): Linear layer of upward connections.
    #     lateral (chainer.links.Linear): Linear layer of lateral connections.

    # .. admonition:: Example

    #     There are several ways to make a StatelessLSTM link.

    #     Let a two-dimensional input array :math:`x`, a cell state array
    #     :math:`h`, and the output array of the previous step :math:`h` be:

    #     >>> x = np.zeros((1, 10), dtype='f')
    #     >>> c = np.zeros((1, 20), dtype='f')
    #     >>> h = np.zeros((1, 20), dtype='f')

    #     1. Give both ``in_size`` and ``out_size`` arguments:

    #         >>> l = L.StatelessLSTM(10, 20)
    #         >>> c_new, h_new = l(c, h, x)
    #         >>> c_new.shape
    #         (1, 20)
    #         >>> h_new.shape
    #         (1, 20)

    #     2. Omit ``in_size`` argument or fill it with ``None``:

    #         The below two cases are the same.

    #         >>> l = L.StatelessLSTM(20)
    #         >>> c_new, h_new = l(c, h, x)
    #         >>> c_new.shape
    #         (1, 20)
    #         >>> h_new.shape
    #         (1, 20)

    #         >>> l = L.StatelessLSTM(None, 20)
    #         >>> c_new, h_new = l(c, h, x)
    #         >>> c_new.shape
    #         (1, 20)
    #         >>> h_new.shape
    #         (1, 20)

    # """

    def call(c, h, x)
      # """Returns new cell state and updated output of LSTM.

      # Args:
      #     c (~chainer.Variable): Cell states of LSTM units.
      #     h (~chainer.Variable): Output at the previous time step.
      #     x (~chainer.Variable): A new batch from the input sequence.

      # Returns:
      #     tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
      #         ``c_new`` represents new cell state, and ``h_new`` is updated
      #         output of LSTM units.

      # """
      unless @upward.w.data
        in_size = x.size / x.shape[0]
        @upward._initialize_params(in_size)
        initialize_params
      end

      lstm_in = @upward.(x)
      lstm_in += @lateral.(h) if h
      unless c
        xp = @xp
        c = Chainer::Variable.new(x.class.zeros(x.shape[0], @state_size))
      end
      Chainer::Functions::Activation::LSTM.lstm(c, lstm_in)
    end
  end


  # """Fully-connected LSTM layer.
  #
  # This is a fully-connected LSTM layer as a chain. Unlike the
  # :func:`~chainer.functions.lstm` function, which is defined as a stateless
  # activation function, this chain holds upward and lateral connections as
  # child links.
  #
  # It also maintains *states*, including the cell state and the output
  # at the previous time step. Therefore, it can be used as a *stateful LSTM*.
  #
  # This link supports variable length inputs. The mini-batch size of the
  # current input must be equal to or smaller than that of the previous one.
  # The mini-batch size of ``c`` and ``h`` is determined as that of the first
  # input ``x``.
  # When mini-batch size of ``i``-th input is smaller than that of the previous
  # input, this link only updates ``c[0:len(x)]`` and ``h[0:len(x)]`` and
  # doesn't change the rest of ``c`` and ``h``.
  # So, please sort input sequences in descending order of lengths before
  # applying the function.
  #
  # Args:
  #     in_size (int): Dimension of input vectors. If it is ``None`` or
  #         omitted, parameter initialization will be deferred until the first
  #         forward data pass at which time the size will be determined.
  #     out_size (int): Dimensionality of output vectors.
  #     lateral_init: A callable that takes ``numpy.ndarray`` or
  #         ``cupy.ndarray`` and edits its value.
  #         It is used for initialization of the lateral connections.
  #         May be ``None`` to use default initialization.
  #     upward_init: A callable that takes ``numpy.ndarray`` or
  #         ``cupy.ndarray`` and edits its value.
  #         It is used for initialization of the upward connections.
  #         May be ``None`` to use default initialization.
  #     bias_init: A callable that takes ``numpy.ndarray`` or
  #         ``cupy.ndarray`` and edits its value
  #         It is used for initialization of the biases of cell input,
  #         input gate and output gate.and gates of the upward connection.
  #         May be a scalar, in that case, the bias is
  #         initialized by this value.
  #         If it is ``None``, the cell-input bias is initialized to zero.
  #     forget_bias_init: A callable that takes ``numpy.ndarray`` or
  #         ``cupy.ndarray`` and edits its value
  #         It is used for initialization of the biases of the forget gate of
  #         the upward connection.
  #         May be a scalar, in that case, the bias is
  #         initialized by this value.
  #         If it is ``None``, the forget bias is initialized to one.
  #
  # Attributes:
  #     upward (~chainer.links.Linear): Linear layer of upward connections.
  #     lateral (~chainer.links.Linear): Linear layer of lateral connections.
  #     c (~chainer.Variable): Cell states of LSTM units.
  #     h (~chainer.Variable): Output at the previous time step.
  class LSTM < LSTMBase
    attr_reader :h, :c

    def initialize(in_size, out_size: nil, **kwargs)
      unless out_size
        out_size = in_size
        in_size = nil
      end
      super(in_size, out_size: out_size, **kwargs)
      reset_state
    end

    def reset_state
      @c = @h = nil
    end

    # Updates the internal state and returns the LSTM outputs.
    # Args:
    #     x (~chainer.Variable): A new batch from the input sequence.
    # Returns:
    #     ~chainer.Variable: Outputs of updated LSTM units.
    def call(x)
      unless @upward.w.data
        in_size = x.size / x.shape[0]
        @upward.initialize_params(in_size)
        initialize_params
      end

      batch = x.shape[0]
      lstm_in = @upward.(x)
      h_rest = nil
      if @h
        h_size = @h.shape[0]
        if batch == 0
          h_rest = @h
        elsif h_size < batch
          msg = ['The batch size of x must be equal to or less than',
                'the size of the previous state h.'].join
          raise TypeError.new(msg)
        elsif h_size > batch
          h_update, h_rest = Chainer::Functions::Array::SplitAxis.split_axis(@h, [batch], 0)
          lstm_in += @lateral.(h_update)
        else
            lstm_in += @lateral.(@h)
        end
      end

      unless @c
        xp = @xp
        dtype = Chainer::Variable === x ? x.dtype : x.class
        @c = Chainer::Variable.new(dtype.zeros(batch, @state_size))
      end
      (@c, y) = Chainer::Functions::Activation::LSTM.lstm(@c, lstm_in)

      if h_rest.nil?
        @h = y
      elsif y.data.shape[0] == 0
        @h = h_rest
      else
        @h = Chainer::Functions::Array::Concat.concat([y, h_rest], axis: 0)
      end
      y
    end
  end
end
