'''
Functions similar to blocks.graph
'''


import logging

import numpy

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.config import config
from blocks.bricks.base import Brick, application
from picklable_itertools.extras import equizip
from blocks.graph import ComputationGraph
from collections import OrderedDict


logger = logging.getLogger(__name__)


class NoiseBrick(Brick):
    """
    A brick to hold parameters introducd by adaptive noise.

    For each model parameter, adaptive noise adds its standard deviations.
    These new parameters will be held by this brick.

    Do not use this brick directly! Its main purpose is to hold noise
    parameters and to wrap the new cost.

    """
    def __init__(self):
        super(NoiseBrick, self).__init__(name='adaptive_noise')
        self.parameters = []
        self.allocated = True
        self.initialized = True

    @application(inputs=['train_cost', 'model_cost',
                         'model_prior_mean', 'model_prior_variance'],
                 outputs=['total_cost'])
    def apply(self, application_call, train_cost, model_cost,
              model_prior_mean, model_prior_variance):
        # We need to add those as auxiliary variables, as they are not
        # used to compute the output, and therefore are lost
        application_call.add_auxiliary_variable(model_prior_mean.copy(),
                                                name='model_prior_mean')
        application_call.add_auxiliary_variable(model_prior_variance.copy(),
                                                name='model_prior_variance')
        total_cost = train_cost + model_cost
        total_cost.name = 'total_cost'
        return total_cost


def __get_name(param):
    brick = None
    for annotation in param.tag.annotations:
        if isinstance(annotation, Brick):
            brick = annotation
            break
    brick_hierarchy = [brick]
    while brick_hierarchy[-1].parents:
        brick_hierarchy.append(brick_hierarchy[-1].parents[0])
    name = "{}.{}".format('/'.join((b.name for b in brick_hierarchy[::-1])),
                          param.name)
    return name


def apply_adaptive_noise(computation_graph,
                         cost,
                         variables,
                         num_examples,
                         parameters=None,
                         init_sigma=1e-6,
                         model_cost_coefficient=1.0,
                         seed=None,
                         gradients=None,
                         ):
    """Add adaptive noise to parameters of a model.

    Each of the given variables will be replaced by a normal
    distribution with learned mean and standard deviation.

    A model cost is computed based on the precision of the the distributions
    associated with each variable. It is added to the given cost used to
    train the model.

    See: A. Graves "Practical Variational Inference for Neural Networks",
         NIPS 2011

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph.
    cost : :class:`~tensor.TensorVariable`
        The cost without weight noise. It should be a member of the
        computation_graph.
    variables : :class:`~tensor.TensorVariable`
        Variables to add noise to.
    num_examples : int
        Number of training examples. The cost of the model is divided by
        the number of training examples, please see
        A. Graves "Practical Variational Inference for Neural Networks"
        for justification
    parameters : list of :class:`~tensor.TensorVariable`
        parameters of the model, if gradients are given the list will not
        be used. Otherwise, it will be used to compute the gradients
    init_sigma : float,
        initial standard deviation of noise variables
    model_cost_coefficient : float,
        the weight of the model cost
    seed : int, optional
        The seed with which
        :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` is initialized,
        is set to 1 by default.
    gradients : dict, optional
        Adaptive weight noise introduces new parameters for which new cost
        and gradients must be computed. Unless the gradients paramter is
        given, it will use theano.grad to get the gradients
    Returns
    -------

    cost : :class:`~tensor.TensorVariable`
        The new cost
    computation_graph : instance of :class:`ComputationGraph`
        new graph with added noise.
    gradients : dict
        a dictionary of gradients for all parameters: the original ones
        and the adaptive noise ones
    noise_brick : :class:~lvsr.graph.NoiseBrick
        the brick that holds all noise parameters and whose .apply method
        can be used to find variables added by adaptive noise
    """
    if not seed:
        seed = config.default_seed
    rng = MRG_RandomStreams(seed)

    try:
        cost_index = computation_graph.outputs.index(cost)
    except ValueError:
        raise ValueError("cost is not part of the computation_graph")

    if gradients is None:
        if parameters is None:
            raise ValueError("Either gradients or parameters must be given")
        logger.info("Taking the cost gradient")
        gradients = dict(equizip(parameters,
                                 tensor.grad(cost, parameters)))
    else:
        if parameters is not None:
            logger.warn("Both gradients and parameters given, will ignore"
                        "parameters")
        parameters = gradients.keys()

    gradients = OrderedDict(gradients)

    log_sigma_scale = 2048.0

    P_noisy = variables  # We will add noise to these
    Beta = []  # will hold means, log_stdev and stdevs
    P_with_noise = []  # will hold parames with added noise

    # These don't change
    P_clean = list(set(parameters).difference(P_noisy))

    noise_brick = NoiseBrick()

    for p in P_noisy:
        p_u = p
        p_val = p.get_value(borrow=True)
        p_ls2 = theano.shared((numpy.zeros_like(p_val) +
                               numpy.log(init_sigma) * 2. / log_sigma_scale
                               ).astype(dtype=numpy.float32))
        p_ls2.name = __get_name(p_u)
        noise_brick.parameters.append(p_ls2)
        p_s2 = tensor.exp(p_ls2 * log_sigma_scale)
        Beta.append((p_u, p_ls2, p_s2))

        p_noisy = p_u + rng.normal(size=p_val.shape) * tensor.sqrt(p_s2)
        p_noisy = tensor.patternbroadcast(p_noisy, p.type.broadcastable)
        P_with_noise.append(p_noisy)

    #  compute the prior mean and variation
    temp_sum = 0.0
    temp_param_count = 0.0
    for p_u, unused_p_ls2, unused_p_s2 in Beta:
        temp_sum = temp_sum + p_u.sum()
        temp_param_count = temp_param_count + p_u.shape.prod()

    prior_u = tensor.cast(temp_sum / temp_param_count, 'float32')

    temp_sum = 0.0
    for p_u, unused_ls2, p_s2 in Beta:
        temp_sum = temp_sum + (p_s2).sum() + (((p_u-prior_u)**2).sum())

    prior_s2 = tensor.cast(temp_sum/temp_param_count, 'float32')

    #  convert everything to use the noisy parameters
    full_computation_graph = ComputationGraph(computation_graph.outputs +
                                              gradients.values())
    full_computation_graph = full_computation_graph.replace(
        dict(zip(P_noisy, P_with_noise)))

    LC = 0.0  # model cost
    for p_u, p_ls2, p_s2 in Beta:
        LC = (LC +
              0.5 * ((tensor.log(prior_s2) - p_ls2 * log_sigma_scale).sum()) +
              1.0 / (2.0 * prior_s2) * (((p_u - prior_u)**2) + p_s2 - prior_s2
                                        ).sum()
              )

    LC = LC / num_examples * model_cost_coefficient

    train_cost = noise_brick.apply(
        full_computation_graph.outputs[cost_index].copy(), LC,
        prior_u, prior_s2)

    gradients = OrderedDict(
        zip(gradients.keys(),
            full_computation_graph.outputs[-len(gradients):]))

    #
    # Delete the gradients form the computational graph
    #
    del full_computation_graph.outputs[-len(gradients):]

    new_grads = {p: gradients.pop(p) for p in P_clean}

    #
    # Warning!!!
    # This only works for batch size 1 (we want that the sum of squares
    # be the square of the sum!
    #
    diag_hessian_estimate = {p: g**2 for p, g in gradients.iteritems()}

    for p_u, p_ls2, p_s2 in Beta:
        p_grad = gradients[p_u]
        p_u_grad = (model_cost_coefficient * (p_u - prior_u) /
                    (num_examples*prior_s2) + p_grad)

        p_ls2_grad = (numpy.float32(model_cost_coefficient *
                                    0.5 / num_examples * log_sigma_scale) *
                      (p_s2/prior_s2 - 1.0) +
                      (0.5*log_sigma_scale) * p_s2 * diag_hessian_estimate[p_u]
                      )
        new_grads[p_u] = p_u_grad
        new_grads[p_ls2] = p_ls2_grad

    return train_cost, full_computation_graph, new_grads, noise_brick
