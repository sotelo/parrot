"""
My new code base in pure Theano/Numpy as a thin wrapper
for simple ops and layers.
Some of the code doesn't belong to me and is gathered from
different sources. (Ask me if the source is not indicated)
"""

import os, sys
sys.path.append(os.getcwd())
sys.path.insert(1, '../')
import lib
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(seed=234)


def uniform(stdev, size):
    """
    uniform distribution with the given stdev and size

    From Ishaan's code:
        https://github.com/igul222/speech
    """
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)

def Linear(
        name,
        input_dims,
        output_dim,
        inputs,
        biases=True,
        initialization=None,
        weightnorm=True,
        just_params=False):
    """
    Compute a linear transform of one or more inputs, optionally with a bias.

    :parameters:
        input_dims: list of ints, or int (if single input); the dimensionality of
                    the input(s).
        output_dim: the dimensionality of the output.
        biases:     whether or not to include a bias term.
        inputs:     a theano variable, or list of variables (if multiple inputs);
                    the inputs to which to apply the transform.
        initialization: one of None, `lecun`, `glorot`, `he`, `glorot_he`, `orthogonal`

    :todo:
        - get arbitrary numpy array as initialization. Check the dims as well.
    """
    if not isinstance(input_dims, list):
        input_dims = [input_dims]
        inputs = [inputs]

    terms = []
    params = []

    for i, (inp, inp_dim) in enumerate(zip(inputs, input_dims)):
        if isinstance(initialization, numpy.ndarray):
            weight_values = initialization
            assert weight_values.shape == (inp_dim, output_dim),\
                'Expecting an ndarray with shape ({}, {}) but got {}'.\
                format(inp_dim, output_dim, initialization.shape)
        elif initialization == 'lecun' or (initialization == None and inp_dim != output_dim):
            weight_values = uniform(numpy.sqrt(1. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'glorot':
            weight_values = uniform(numpy.sqrt(2./(inp_dim+output_dim)), (inp_dim, output_dim))
        elif initialization == 'he':
            weight_values = uniform(numpy.sqrt(2. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'glorot_he':
            weight_values = uniform(numpy.sqrt(4./(inp_dim+output_dim)), (inp_dim, output_dim))
        elif initialization == 'orthogonal' or (initialization == None and inp_dim == output_dim):
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are supported.")
                flat_shape = (shape[0], numpy.prod(shape[1:]))
                # TODO: why normal and not uniform?
                a = numpy.random.normal(0.0, 1.0, flat_shape)
                u, _, v = numpy.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(theano.config.floatX)
            weight_values = sample((inp_dim, output_dim))
        else:
            raise Exception("Invalid initialization ({})!"\
                    .format(repr(initialization)))

        weight = lib.param(
            name + '.W'+str(i),
            weight_values
        )
        params.append(weight)

        if weightnorm:
            norm_values = numpy.linalg.norm(weight_values, axis=0)
            norms = lib.param(
                name + '.g'+str(i),
                norm_values
            )
            params.append(norms)

            normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
            prepared_weight = normed_weight
        else:
            prepared_weight = weight
        terms.append(T.dot(inp, prepared_weight))

    if biases:
        layer_biases = lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        )
        params.append(layer_biases)
        terms.append(layer_biases)

    if just_params:
        return params
    # otherwise, comlete/add to the computation graph
    out = reduce(lambda a,b: a+b, terms)
    out.name = name + '.output'
    return out

def Batchnorm(
    name,
    input_dim,
    inputs,
    stepwise=False,
    axes=None,
    wrt=None,
    i_gamma=None,
    i_beta=None):
    """
    From Ishaan's repo
    """
    if wrt is None:
        wrt = inputs

    if axes is not None:
        means = wrt.mean(axis=axes, keepdims=True)
        variances = wrt.var(axis=axes, keepdims=True)
    # elif stepwise:
    #     means = wrt.mean(axis=1, keepdims=True)
    #     variances = wrt.var(axis=1, keepdims=True)
    else:
        means = wrt.reshape((-1, input_dim)).mean(axis=0)
        variances = wrt.reshape((-1, input_dim)).var(axis=0)

    if i_gamma is None:
        i_gamma = lib.floatX(0.1) * numpy.ones(input_dim, dtype=theano.config.floatX)

    if i_beta is None:
        i_beta = numpy.zeros(input_dim, dtype=theano.config.floatX)

    gamma = lib.param(
        name + '.gamma',
        i_gamma
    )

    beta = lib.param(
        name + '.beta',
        i_beta
    )

    stdevs = T.sqrt(variances + lib.floatX(1e-6))

    stdevs.name = name+'.stdevs'
    means.name = name+'.means'

    # return (((inputs - means) / stdevs) * gamma) + beta
    if axes is not None:
        dimshuffle_pattern = [
            'x' if i in axes else 0
            for i in xrange(inputs.ndim)
        ]
        return T.nnet.bn.batch_normalization(
            inputs,
            gamma.dimshuffle(*dimshuffle_pattern),
            beta.dimshuffle(*dimshuffle_pattern),
            means,
            stdevs,
            mode='low_mem'
        )
    else:
        return T.nnet.bn.batch_normalization(
            inputs,
            gamma.dimshuffle('x',0),
            beta.dimshuffle('x',0),
            means.dimshuffle('x',0),
            stdevs.dimshuffle('x',0),
            mode='low_mem'
        )

def ReLULayer(name, input_dim, output_dim, inputs, batchnorm=False):
    output = Linear(
        name+'.Linear',
        input_dims=input_dim,
        output_dim=output_dim,
        inputs=inputs,
        initialization='glorot_he',
        biases=(not batchnorm),
        weightnorm=False
    )

    if batchnorm:
        output = Batchnorm(
            name+'.BN',
            input_dim=output_dim,
            inputs=output
        )

    output = T.nnet.relu(output)

    return output

def MLP(name, input_dim, hidden_dim, output_dim, n_layers, inputs, batchnorm=True):
    if n_layers < 3:
        raise Exception("An MLP with <3 layers isn't an MLP!")

    output = ReLULayer(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        inputs=inputs,
        batchnorm=batchnorm
    )

    for i in xrange(1, n_layers-2):
        output = ReLULayer(
            name+'.Hidden'+str(i),
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            inputs=output,
            batchnorm=batchnorm
        )

    return Linear(
        name+'.Output',
        hidden_dim,
        output_dim,
        output,
        initialization='glorot',
        weightnorm=False
    )

def Embedding(name, n_symbols, output_dim, indices):
    vectors = lib.param(
        name,
        numpy.random.randn(
            n_symbols,
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = [
        indices.shape[i]
        for i in xrange(indices.ndim)
    ] + [output_dim]

    return vectors[indices.flatten()].reshape(output_shape)

def softmax_and_sample(logits, temperature=1.):
    """
    :temperature: default 1.
    For high temperatures (temperature -> +Inf), all actions have nearly the same
    probability and the lower the temperature, the more expected rewards affect
    the probability. For a low temperature (temperature -> 0+), the probability of
    the action with the highest expected reward (max operation) tends to 1.
    """
    temperature = lib.floatX(temperature)
    ZEROX = lib.floatX(0.)
    assert temperature >= ZEROX, "`temperature` should be a non-negative value!"
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))

    if temperature == ZEROX:
        # Get max instead of (biased) sample.
        # Equivalent to directly get the argmax but with this it's easier to
        # extract the probabilities later on too.
        samples = T.nnet.softmax(flattened_logits)
    else: # > 0
        flattened_logits /= temperature
        samples = T.cast(
            srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
            theano.config.floatX
        )
    samples = samples.reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)

def softmax_and_argmax(logits):
    return softmax_and_sample(logits, temperature=0)

def __Recurrent(name, hidden_dims, step_fn, inputs, non_sequences=[], h0s=None):
    if not isinstance(inputs, list):
        inputs = [inputs]

    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]

    if h0s is None:
        h0s = [None]*len(hidden_dims)

    for i in xrange(len(hidden_dims)):
        if h0s[i] is None:
            h0_unbatched = lib.param(
                name + '.h0_' + str(i),
                numpy.zeros((hidden_dims[i],), dtype=theano.config.floatX)
            )
            num_batches = inputs[0].shape[1]
            h0s[i] = T.alloc(h0_unbatched, num_batches, hidden_dims[i])

        h0s[i] = T.patternbroadcast(h0s[i], [False] * h0s[i].ndim)

    outputs, _ = theano.scan(
        step_fn,
        sequences=inputs,
        outputs_info=h0s,
        non_sequences=non_sequences
    )

    return outputs

def __GRUStep(
        name,
        input_dim,
        hidden_dim,
        current_input,
        last_hidden,
        weightnorm=True):
    """
    CAUTION:
        Not for stand-alone usage. It is defined here (instead of
        inside VanillaRNN function) to not clutter the code.

    Note:
        No 'Output' gate. 'Input' and 'Forget' gates coupled by an update
        gate z and the reset gate r is applied directly to the previous hidden
        state. Thus, the responsibility of the reset gate in a LSTM is really
        split up into both r and z.

    Gates:
        z = sigm(X_t*U^z + S_{t-1}*W^z)
        r = sigm(X_t*U^r + S_{t-1}*W^r)
    Candidate:
        h = tanh(X_t*U^h + (S_{t-1}.r)*W^h)
        S_t = (1 - z).h + z.S_{t-1}
    """
    # x_t*(U^z, U^r, U^h)
    # Also contains biases
    processed_input = lib.ops.Linear(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        current_input,
        weightnorm=weightnorm
    )

    gates = T.nnet.sigmoid(
        lib.ops.Linear(
            name+'.Recurrent_Gates',
            hidden_dim,
            2 * hidden_dim,
            last_hidden,
            biases=False,
            weightnorm=weightnorm
        ) + processed_input[:, :2*hidden_dim]
    )

    update = gates[:, :hidden_dim]
    reset  = gates[:, hidden_dim:]

    scaled_hidden = reset * last_hidden

    candidate = T.tanh(
        lib.ops.Linear(
            name+'.Recurrent_Candidate',
            hidden_dim,
            hidden_dim,
            scaled_hidden,
            biases=False,
            initialization='orthogonal',
            weightnorm=weightnorm
        ) + processed_input[:, 2*hidden_dim:]
    )

    one = lib.floatX(1.0)
    return (update * candidate) + ((one - update) * last_hidden)

def LowMemGRU(
        name,
        input_dim,
        hidden_dim,
        inputs,
        h0=None,
        mask=None,
        weightnorm=True):
    """
    :todo:
        - Right now masking is not implemented and passing that arguement is
          not doing anything.

    :usage:
        >>> TODO
    """
    inputs = inputs.dimshuffle(1,0,2)

    #if mask is None:
    #    mask =
    def step(current_input, last_hidden):
        return __GRUStep(
            name+'.Step',
            input_dim,
            hidden_dim,
            current_input,
            last_hidden,
            weightnorm=weightnorm
        )

    if h0 is None:
        h0s = None
    else:
        h0s = [h0]

    out = __Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        inputs,
        h0s=h0s
    )

    out = out.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def __VanillaRNNstep(
        name,
        input_dim,
        hidden_dim,
        current_inp,
        last_hidden,
        weightnorm=True):
    """
    CAUTION:
        Not for stand-alone usage. It is defined here (instead of
        inside VanillaRNN function) to not clutter the code.

    :todo:
        - Implement!
        - Test!
    """
    # S_t = tanh(U*X_t+W*S_{t-1})
    raise NotImplementedError

def __LSTMStep(
        name,
        input_dim,
        hidden_dim,
        current_input,
        last_hidden,
        weightnorm=True,
        inp_bias_init=0.,
        forget_bias_init=3.,
        out_bias_init=0.,
        g_bias_init=0.):
    """
    CAUTION:
        Not for stand-alone usage. It is defined here (instead of
        inside LSTM function) to not clutter the code.

    Gates:
        i = sigm(X_t*U^i + S_{t-1}*W^i + b^i)
        f = sigm(X_t*U^f + S_{t-1}*W^f + b^f)
        o = sigm(X_t*U^o + S_{t-1}*W^o + b^o)
    Candidate/internal mempry/cell state and hidden state:
        g = tanh(X_t*U^g + S_{t-1}*W^g + b^g)
        c_t = c_{t-1}.f + g.i
    State:
        S_t = tanh(c_t).o
    last_hidden:
        dim: (2*hidden_dim)
        S_{t-1} = last_hidden[:hidden_dim]
        c_{t-1} = last_hidden[hidden_dim:]

    Note:
        Forget gate bias initalizations with large positive values (1. to 5.)
        is shown to be beneficial for learning an/or modeling long-term
        dependencies.
        sigmoid([0., 1., 2., 3., 5.]) = [.5, .73, .88, 95., .99]
    See:
        http://www.felixgers.de/papers/phd.pdf
        http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

    :todo:
        - Better initializations, especially for the weight matrices.
        - Fix the 'concatenation' to use instead of T.concatention
    """
    # X_t*(U^i, U^f, U^o, U^g)
    processed_input = lib.ops.Linear(
        name+'.Input',
        input_dim,
        4 * hidden_dim,
        current_input,
        biases=False,
        weightnorm=weightnorm
    )

    # last_hidden is [batch size, S_{t-1};c_{t-1}]
    s_tm1 = last_hidden[:, :hidden_dim]
    c_tm1 = last_hidden[:, hidden_dim:]
    # S_{t-1}*(W^i, W^f, W^o, W^g)
    processed_last_hidden = lib.ops.Linear(
        name+'.Recurrent_Gates',
        hidden_dim,
        4 * hidden_dim,
        s_tm1,
        biases=False,
        weightnorm=weightnorm
    )

    # All the fancy bias initialization: b^i, b^f, b^o, b^g
    gate_bias_inits = numpy.zeros((4*hidden_dim,), dtype=theano.config.floatX)
    gate_bias_inits[:hidden_dim]               = inp_bias_init
    gate_bias_inits[hidden_dim:2*hidden_dim]   = forget_bias_init
    gate_bias_inits[2*hidden_dim:3*hidden_dim] = out_bias_init
    gate_bias_inits[3*hidden_dim:]             = g_bias_init
    biases = lib.param(name + '.b', gate_bias_inits)

    pre_gates  = processed_input + processed_last_hidden  # 4*dim
    pre_gates += biases  # 4*dim
    gates      = T.nnet.sigmoid(pre_gates[:, :3*hidden_dim])  # 3*dim

    inp    = gates[:, :hidden_dim]  # dim
    forget = gates[:, hidden_dim:2*hidden_dim]  # dim
    out    = gates[:, 2*hidden_dim:]  # dim

    g = T.tanh(pre_gates[:, 3*hidden_dim:])  # dim

    # internal memory/cell state
    c_t = c_tm1 * forget + g * inp  # dim
    # hidden state
    s_t = T.tanh(c_t) * out  # dim
    # TODO: Again, problem with concatenating tensors with (False, False)
    # broadcast pattern. If slow down as a result of transferring to CPU for
    # concatenation is not high, keep it this way.
    hidden_state = T.concatenate([s_t, c_t], axis=-1) # 2*dim, axis=1
    return hidden_state

def LowMemLSTM(
        name,
        input_dim,
        hidden_dim,
        inputs,
        h0=None,
        mask=None,
        weightnorm=True):
    """
    Note:
        A LSTM layer is just another way to compute a hidden state. Previously,
        we computed the hidden state as s_t = tanh(Ux_t + Ws_{t-1}). The inputs
        to this unit were x_t, the current input at step t, and s_{t-1}, the
        previous hidden state.  The output was a new hidden state s_t. A LSTM
        unit does the exact same thing, just in a different way!
        The LSTM's output is typically taken to be S_t (or h_t), and c_t is not
        exposed. The forget gate allows the LSTM to easily reset the value of
        the cell.

    :usage:
        >>> TODO

    :todo:
        - Right now masking is not implemented and passing that arguement is
          not doing anything.
    """
    inputs = inputs.dimshuffle(1,0,2)

    #if mask is None:
    #    mask =
    def step(current_input, last_hidden):
        return __LSTMStep(
            name+'.Step',
            input_dim,
            hidden_dim,
            current_input,
            last_hidden,
            weightnorm=weightnorm
        )

    if h0 is None:
        h0s = None
    else:
        h0s = [h0]

    out = __Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        inputs,
        h0s=h0s
    )

    out = out.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def stackedGRU(
        name,
        n_rnn,
        input_dim,
        hidden_dim,
        inputs,
        h0,
        weightnorm,
        skip_conn):
    """
    Note:
        Hard-coded stacked GRU. Number of GRUs should be smaller than 6.
        Also handles skip connections which is to be added in the 'automatic'
        version of this function.
        For n_rnn = 1, skip connection is not defined. Just accepts
        skip_conn == False and saves computation time.

        h0 should have the shape of: (:, n_rnn, hidden_dim)

    See:
        'Generating sequences with recurrent neural networks' by A. Graves

    :usage:
        >>> TODO
    """
    assert n_rnn in xrange(1, 6), "n_rnn should be in [1,2,3,4,5]"
    assert not(n_rnn == 1 and skip_conn == True),\
            "Single layer RNN cannot have skip connections"

    # 1st layer GRU
    gru1_inp = inputs
    gru1_inp_dim = input_dim
    gru1 = LowMemGRU(name+'1',
                     gru1_inp_dim,
                     hidden_dim,
                     gru1_inp,
                     h0=h0[:, 0],
                     weightnorm=weightnorm)
    if not skip_conn:
        out = gru1
    else:
        # Just note that a single layer RNN doesn't have skip connections.
        # If you reach here, it means it's more than 1 layer and with enabled
        # skip connection.
        # Among output skip connections, this one only includes biases. (Look
        # at Graves' paper)
        out = Linear(name+'.outskip1y',
                     hidden_dim,
                     hidden_dim,
                     gru1,
                     biases=True,
                     initialization='he',
                     weightnorm=weightnorm)
    last_hiddens = [gru1[:, -1]]

    extra_name = '+inpskip' if skip_conn else ''

    # 2nd layer RNN
    if n_rnn > 1:
        if not skip_conn:
            gru2_inp = gru1
            gru2_inp_dim = hidden_dim
        else:
            # TODO: Find out why concatenate did not worked here for two inputs
            # of (False, False, False) broadcastable. concatenate([gru1,
            # inputs], axis=2).broadcastable is (False, False, False, False)???
            gru2_inp = T.concatenate([gru1, inputs], axis=-1) # axis 2
            gru2_inp_dim = hidden_dim + input_dim
        gru2 = LowMemGRU(name+'2'+extra_name,
                         gru2_inp_dim,
                         hidden_dim,
                         gru2_inp,
                         h0=h0[:, 1],
                         weightnorm=weightnorm)
        if not skip_conn:
            out = gru2
        else:
            out += Linear(name+'.outskip2y',
                          hidden_dim,
                          hidden_dim,
                          gru2,
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(gru2[:, -1])

    # 3rd layer RNN
    if n_rnn > 2:
        if not skip_conn:
            gru3_inp = gru2
            gru3_inp_dim = hidden_dim
        else:
            gru3_inp = T.concatenate([gru2, inputs], axis=-1) # axis 2
            gru3_inp_dim = hidden_dim + input_dim
        gru3 = LowMemGRU(name+'3'+extra_name,
                         gru3_inp_dim,
                         hidden_dim,
                         gru3_inp,
                         h0=h0[:, 2],
                         weightnorm=weightnorm)
        if not skip_conn:
            out = gru3
        else:
            out += Linear(name+'.outskip3y',
                          hidden_dim,
                          hidden_dim,
                          gru3,
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(gru3[:, -1])

    # 4th layer RNN
    if n_rnn > 3:
        if not skip_conn:
            gru4_inp = gru3
            gru4_inp_dim = hidden_dim
        else:
            gru4_inp = T.concatenate([gru3, inputs], axis=-1) # axis 2
            gru4_inp_dim = hidden_dim + input_dim
        gru4 = LowMemGRU(name+'4'+extra_name,
                         gru4_inp_dim,
                         hidden_dim,
                         gru4_inp,
                         h0=h0[:, 3],
                         weightnorm=weightnorm)
        if not skip_conn:
            out = gru4
        else:
            out += Linear(name+'.outskip4y',
                          hidden_dim,
                          hidden_dim,
                          gru4,
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(gru4[:, -1])

    # 5th layer RNN
    if n_rnn > 4:
        if not skip_conn:
            gru5_inp = gru4
            gru5_inp_dim = hidden_dim
        else:
            gru5_inp = T.concatenate([gru4, inputs], axis=-1) # axis 2
            gru5_inp_dim = hidden_dim + input_dim
        gru5 = LowMemGRU(name+'5'+extra_name,
                         gru5_inp_dim,
                         hidden_dim,
                         gru5_inp,
                         h0=h0[:, 4],
                         weightnorm=weightnorm)
        if not skip_conn:
            out = gru5
        else:
            out += Linear(name+'.outskip5y',
                          hidden_dim,
                          hidden_dim,
                          gru5,
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(gru5[:, -1])

    last_hiddens = T.stack(last_hiddens, axis=1)
    return out, last_hiddens

# TODO: Haven't tried and compared it yet. For now stick to the manual
# implementation of stacked RNN.
#def stackedGRU(name, n_rnn, input_dim, hidden_dim, inputs, h0, weightnorm=True):
#    """
#    Not a generalized stacked GRU, yet. hidden_dim will be applied to all
#    the RNNs, should be one scalar.
#    """
#    gru_inp = inputs
#    gru_inp_dim = input_dims
#    last_hiddens = []
#    for i in xrange(n_rnn):
#        gru_out = LowMemGRU(name+str(i+1),
#                            gru_inp_dim,
#                            hidden_dim,
#                            gru_inp,
#                            h0=h0[:, i],
#                            weightnorm=weightnorm)
#        last_hiddens.append(gru_out[:, -1])
#        gru_inp_dim = hidden_dims
#        gru_inp = gru_out
#
#    return gru_out, last_hiddens
##    gru1_inp = frames
##    # (batch_size, n_frames, frame_size+n_global_features)
##    gru1 = lib.ops.LowMemGRU('FrameLevel.GRU1',
##                              FRAME_SIZE,
##                              DIM,
##                              gru1_inp,
##                              h0=h0[:, 0])
##
##    gru2_inp = gru1
##    gru2 = lib.ops.LowMemGRU('FrameLevel.GRU2',
##                             DIM,
##                             DIM,
##                             gru2_inp,
##                             h0=h0[:, 1])
##
##    gru3_inp = gru2
##    gru3 = lib.ops.LowMemGRU('FrameLevel.GRU3',
##                             DIM,
##                             DIM,
##                             gru3_inp,
##                             h0=h0[:, 2])

def stackedLSTM(
        name,
        n_rnn,
        input_dim,
        hidden_dim,
        inputs,
        h0,
        weightnorm,
        skip_conn):
    """
    Note:
        Hard-coded stacked LSTM. Number of LSTMs should be smaller than 6.
        Also handles skip connections which is to be added in the 'automatic'
        version of this function.
        For n_rnn = 1, skip connection is not defined. Just accepts
        skip_conn == False and saves computation time.

        h0 should have the shape of: (:, n_rnn, 2*hidden_dim)

    See:
        'Generating sequences with recurrent neural networks' by A. Graves

    :usage:
        >>> TODO
    """
    assert n_rnn in xrange(1, 6), "n_rnn should be in [1,2,3,4,5]"
    assert not(n_rnn == 1 and skip_conn == True),\
            "Single layer RNN cannot have skip connections"

    # 1st layer LSTM
    lstm1_inp = inputs
    lstm1_inp_dim = input_dim
    lstm1 = LowMemLSTM(name+'1',
                       lstm1_inp_dim,
                       hidden_dim,
                       lstm1_inp,
                       h0=h0[:, 0],
                       weightnorm=weightnorm)

    if not skip_conn:
        out = lstm1[:, :, :hidden_dim]
    else:
        # Just note that a single layer RNN doesn't have skip connections.
        # If you reach here, it means it's more than 1 layer and with enabled
        # skip connection.
        # Among output skip connections, this one only includes biases. (Look
        # at Graves' paper)
        out = Linear(name+'1.outskip1y',
                     hidden_dim,
                     hidden_dim,
                     lstm1[:, :, :hidden_dim],
                     biases=True,
                     initialization='he',
                     weightnorm=weightnorm)
    last_hiddens = [lstm1[:, -1]]

    extra_name = '+inpskip' if skip_conn else ''

    # 2nd layer RNN
    if n_rnn > 1:
        if not skip_conn:
            lstm2_inp = lstm1[:, :, :hidden_dim]
            lstm2_inp_dim = hidden_dim
        else:
            # TODO: Find out why concatenate did not worked here for two inputs
            # of (False, False, False) broadcastable. concatenate([gru1,
            # inputs], axis=2).broadcastable is (False, False, False, False)???
            lstm2_inp = T.concatenate([lstm1[:, :, :hidden_dim], inputs], axis=-1) # axis 2
            lstm2_inp_dim = hidden_dim + input_dim
        lstm2 = LowMemLSTM(name+'2'+extra_name,
                           lstm2_inp_dim,
                           hidden_dim,
                           lstm2_inp,
                           h0=h0[:, 1],
                           weightnorm=weightnorm)
        if not skip_conn:
            out = lstm2[:, :, :hidden_dim]
        else:
            out += Linear(name+'2.outskip2y',
                          hidden_dim,
                          hidden_dim,
                          lstm2[:, :, :hidden_dim],
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(lstm2[:, -1])

    # 3rd layer RNN
    if n_rnn > 2:
        if not skip_conn:
            lstm3_inp = lstm2[:, :, :hidden_dim]
            lstm3_inp_dim = hidden_dim
        else:
            lstm3_inp = T.concatenate([lstm2[:, :, :hidden_dim], inputs], axis=-1) # axis 2
            lstm3_inp_dim = hidden_dim + input_dim
        lstm3 = LowMemLSTM(name+'3'+extra_name,
                           lstm3_inp_dim,
                           hidden_dim,
                           lstm3_inp,
                           h0=h0[:, 2],
                           weightnorm=weightnorm)
        if not skip_conn:
            out = lstm3[:, :, :hidden_dim]
        else:
            out += Linear(name+'3.outskip3y',
                          hidden_dim,
                          hidden_dim,
                          lstm3[:, :, :hidden_dim],
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(lstm3[:, -1])

    # 4th layer RNN
    if n_rnn > 3:
        if not skip_conn:
            lstm4_inp = lstm3[:, :, :hidden_dim]
            lstm4_inp_dim = hidden_dim
        else:
            lstm4_inp = T.concatenate([lstm3[:, :, :hidden_dim], inputs], axis=-1) # axis 2
            lstm4_inp_dim = hidden_dim + input_dim
        lstm4 = LowMemLSTM(name+'4'+extra_name,
                           lstm4_inp_dim,
                           hidden_dim,
                           lstm4_inp,
                           h0=h0[:, 3],
                           weightnorm=weightnorm)
        if not skip_conn:
            out = lstm4[:, :, :hidden_dim]
        else:
            out += Linear(name+'4.outskip4y',
                          hidden_dim,
                          hidden_dim,
                          lstm4[:, :, :hidden_dim],
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(lstm4[:, -1])

    # 5th layer RNN
    if n_rnn > 4:
        if not skip_conn:
            lstm5_inp = lstm4[:, :, :hidden_dim]
            lstm5_inp_dim = hidden_dim
        else:
            lstm5_inp = T.concatenate([lstm4[:, :, :hidden_dim], inputs], axis=-1) # axis 2
            lstm5_inp_dim = hidden_dim + input_dim
        lstm5 = LowMemLSTM(name+'5'+extra_name,
                           lstm5_inp_dim,
                           hidden_dim,
                           lstm5_inp,
                           h0=h0[:, 4],
                           weightnorm=weightnorm)
        if not skip_conn:
            out = lstm5[:, :, :hidden_dim]
        else:
            out += Linear(name+'5.outskip5y',
                          hidden_dim,
                          hidden_dim,
                          lstm5[:, :, :hidden_dim],
                          biases=False,
                          initialization='he',
                          weightnorm=weightnorm)
        last_hiddens.append(lstm5[:, -1])

    last_hiddens = T.stack(last_hiddens, axis=1)
    return out, last_hiddens

def gaussian_nll(x, mus, sigmas):
    """
    NLL for Multivariate Normal with diagonal covariance matrix
    See:
        wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
    where \Sigma = diag(s_1^2,..., s_n^2).

    x, mus, sigmas all should have the same shape.
    sigmas (s_1,..., s_n) should be strictly positive.
    Results in output shape of similar but without the last dimension.
    """
    nll = lib.floatX(numpy.log(2. * numpy.pi))
    nll += 2. * T.log(sigmas)
    nll += ((x - mus) / sigmas) ** 2.
    nll = nll.sum(axis=-1)
    nll *= lib.floatX(0.5)
    return nll

def GMM_nll(x, mus, sigmas, mix_weights):
    """
    D is dimension of each observation (e.g. frame_size) for each component
    (multivariate Normal with diagonal covariance matrix)
    See `gaussian_nll`

    x : (batch_size, D)
    mus : (batch_size, D, num_gaussians)
    sigmas : (batch_size, D, num_gaussians)
    mix_weights : (batch_size, num_gaussians)
    """
    x = x.dimshuffle(0, 1, 'x')

    # Similar to `gaussian_nll`
    ll_component_wise = lib.floatX(numpy.log(2. * numpy.pi))
    ll_component_wise += 2. * T.log(sigmas)
    ll_component_wise += ((x - mus) / sigmas) ** 2.
    ll_component_wise = ll_component_wise.sum(axis=1)  # on FRAME_SIZE
    ll_component_wise *= lib.floatX(-0.5)  # LL not NLL

    # Now ready to take care of weights of each component
    # Simply applying exp could potentially cause inf/NaN.
    # Look up LogSumExp trick, Softmax in theano, or this:
    # hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    weighted_ll = ll_component_wise + T.log(mix_weights)
    ll_max = T.max(weighted_ll, axis=1, keepdims=True)
    nll = T.log(T.sum(T.exp(weighted_ll - ll_max), axis=1, keepdims=True))
    nll += ll_max
    nll = -nll.sum(axis=1)
    return nll

def GMM_sample(mus, sigmas, mix_weights):
    """
    First, sample according to the prior mixing probabilities
    to choose the component density.
    Second, draw sample from that density

    Inspired by implementation in `cle`
    """
    chosen_component = \
        T.argmax(
            srng.multinomial(pvals=mix_weights),
            axis=1)
    selected_mus = mus[T.arange(mus.shape[0]), :, chosen_component]
    selected_sigmas = sigmas[T.arange(sigmas.shape[0]), :, chosen_component]
    sample = srng.normal(size=selected_mus.shape,
                                avg=0.,
                                std=1.)
    sample *= selected_sigmas
    sample += selected_mus
    return sample, selected_mus, selected_sigmas, chosen_component

def concatenate(tensor_list, axis=0):
    """
    Note:
        Alternative implementation of `theano.tensor.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.tensor.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.

    From Cho's code here:
        https://github.com/nyu-dl/dl4mt-tutorial/blob/master/session2/nmt.py#L115

    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)

    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.

    :returns:
        - out : tensor
            the concatenated tensor expression.

    :todo:
        - Doesn't work properly when input tensors have all False broadcastable
          patterns. (E.g. LowMemLSTM and LowMemGRU)
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

def dropout_layer(state_before, use_noise, trng):
    """
    :todo:
        - Fix according to _param
        - Test!

    From Cho's code here:
        https://github.com/nyu-dl/dl4mt-tutorial/blob/master/session2/nmt.py#L45
    """
    proj = tensor.switch(
        use_noise,
        # for training
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        # for validation/sampling
        state_before * 0.5)
    return proj

def extend_middle_dim(_2D, num):
    """
    Gets a 2D tensor (A, B), outputs a 3D tensor (A, num, B)
    :usage:
        >>> TODO
    """
    rval = _2D.dimshuffle((0, 'x', 1))
    rval = T.alloc(rval, rval.shape[0], num, rval.shape[2])
    return rval

def T_one_hot(inp_tensor, n_classes):
    """
    :todo:
        - Implement other methods from here:
        - Compare them for speed-wise for different sizes
        - Implement N_one_hot for Numpy version, with speed tests.

    Theano one-hot (1-of-k) from an input tensor of indecies.
    If the indecies are of the shape (a0, a1, ..., an) the output
    shape would be (a0, a1, ..., a2, n_classes).

    :params:
        - inp_tensor: any theano tensor with dtype int* as indecies and all of
                      them between [0, n_classes-1].
        - n_classes: number of classes which determines the output size.

    :usage:
        >>> idx = T.itensor3()
        >>> idx_val = numpy.array([[[0,1,2,3],[4,5,6,7]]], dtype='int32')
        >>> one_hot = T_one_hot(t, 8)
        >>> one_hot.eval({idx:idx_val})
        >>> print out
        array([[[[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]],
        [[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]]]])
        >>> print idx_val.shape, out.shape
        (1, 2, 4) (1, 2, 4, 8)
    """
    flattened = inp_tensor.flatten()
    z = T.zeros((flattened.shape[0], n_classes), dtype=theano.config.floatX)
    one_hot = T.set_subtensor(z[T.arange(flattened.shape[0]), flattened], 1)
    out_shape = [inp_tensor.shape[i] for i in xrange(inp_tensor.ndim)] + [n_classes]
    one_hot = one_hot.reshape(out_shape)
    return one_hot

def gated_non_linerity(x):
    gate = x[:,::2]
    val = x[:,1::2]
    return T.tanh(val) * T.nnet.sigmoid(gate)

def dil_conv_1D(
    input_,
    output_dim,
    input_dim,
    filter_size,
    dilation = 1,
    non_linearity = 'gated',
    name = None,
    init = 'glorot'
    ):
    """
    :params:
        - inp: theano tensor of shape (batch_size, timesteps, input_dim)

    :output:
        - output : theano tensor of shape (batch_size, timesteps, output_dim)
    """

    assert(name is not None)
    #assert("0.8" not in theano.__version__)
    assert(filter_size == 2)

    #TODO: Remove redundancy, use conv1d instead

    import lasagne

    inp = input_.dimshuffle(0,2,1,'x')

    if init == 'glorot':
        initializer = lasagne.init.GlorotUniform()
    elif init == 'he':
        initializer = lasagne.init.HeUniform()

    dilation_ = (dilation, 1)


    if non_linearity == 'gated':
        num_filters = 2*output_dim
    else:
        num_filters = output_dim

    W_shape = (num_filters, input_dim, filter_size, 1)
    bias_shape = (num_filters,)

    W = lib.param(name+".W", initializer.sample(W_shape))
    b = lib.param(name+".b", lasagne.init.Constant(0.).sample(bias_shape))


    W1x1_shape = (2*output_dim, output_dim, 1, 1)
    W1x1 = lib.param(name+".W1x1", initializer.sample(W1x1_shape))

    conv_out = T.nnet.conv2d(
                    inp,  W,
                    filter_flip= False,
                    border_mode = 'valid',
                    filter_dilation = dilation_
                )

    conv_out = conv_out + b[None,:,None, None]

    if non_linearity == 'gated':
        activation = gated_non_linerity
    elif non_linearity == 'relu':
        activation = T.nnet.relu
    elif non_linearity == 'elu':
        activation = lambda x : T.switch( x >= 0, x, T.exp(x) - floatX(1.))
    elif non_linearity == 'identity':
        activation = lambda x: x
    else:
        raise NotImplementedError("{} non-linearity not implemented!".format(non_linearity))

    out_temp = activation(conv_out)

    output = T.nnet.conv2d(
                    out_temp,  W1x1,
                    border_mode = 'valid'
                )

    output = output.reshape((output.shape[0], output.shape[1], output.shape[2]))
    output = output.dimshuffle(0,2,1)

    if input_dim == output_dim:
        return output[:,:,::2] + input_[:,dilation:], output[:,:,1::2]
    else:
        return output[:,:,::2], output[:,:,1::2]

def conv1d(
    name,
    input,
    input_dim,
    output_dim,
    filter_size,
    init = 'glorot',
    non_linearity = 'relu',
    bias = True
    ):

    import lasagne

    inp = input.dimshuffle(0,2,1,'x')

    if init == 'glorot':
        initializer = lasagne.init.GlorotUniform()
    elif init == 'he':
        initializer = lasagne.init.HeUniform()

    if non_linearity == 'gated':
        num_filters = 2*output_dim
    else:
        num_filters = output_dim

    W_shape = (num_filters, input_dim, filter_size, 1)

    if bias:
        bias_shape = (num_filters,)

    W = lib.param(name+".W", initializer.sample(W_shape))

    if bias:
        b = lib.param(name+".b", lasagne.init.Constant(0.).sample(bias_shape))

    conv_out = T.nnet.conv2d(
                    inp,  W,
                    filter_flip= False,
                    border_mode = 'valid'
                )

    if bias:
        conv_out = conv_out + b[None,:,None, None]

    if non_linearity == 'gated':
        activation = gated_non_linerity
    elif non_linearity == 'relu':
        activation = T.nnet.relu
    elif non_linearity == 'elu':
        activation = lambda x : T.switch( x >= 0., x, T.exp(x) - floatX(1.))
    elif non_linearity == 'identity':
        activation = lambda x: x
    else:
        raise NotImplementedError("{} non-linearity not implemented!".format(non_linearity))

    output = conv_out

    output = output.reshape((output.shape[0], output.shape[1], output.shape[2]))
    output = output.dimshuffle(0,2,1)

    return output
