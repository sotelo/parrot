import numpy

import theano
import logging
from theano import tensor

from collections import OrderedDict

from blocks.algorithms import StepRule
from blocks.utils import shared_floatx
from blocks.theano_expressions import l2_norm


class BurnIn(StepRule):
    """Zeroes the updates until a number of steps is performed.


    Parameters
    ----------
    num_steps : int, default 0
        The number of steps during which updates are disabled

    Attributes
    ----------
    num_steps : :class:`.tensor.TensorSharedVariable`
        The remaining number of burn_in steps

    """
    def __init__(self, num_steps=0):
        self.num_steps = theano.shared(num_steps)

    def compute_steps(self, previous_steps):
        multiplier = tensor.switch(self.num_steps <= 0,
                                   1, 0)
        steps = OrderedDict(
            (parameter, step * multiplier)
            for parameter, step in previous_steps.items())
        return steps, [(self.num_steps, tensor.maximum(0, self.num_steps - 1))]


class AdaptiveStepClipping(StepRule):
    """Tracks the magnitude of the gradient and adaptively rescales it.

    When the previous steps are the gradients, this step rule performs
    gradient clipping described in [JCh2014]_.

    .. [JCh2014] JCH NIPS Workshop TODO

    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the step. The step
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.

    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.

    """
    def __init__(self, initial_threshold=1.0, stdevs=4, decay=0.96,
                 clip_to_mean=True, quick_variance_convergence=True,
                 **kwargs):
        super(AdaptiveStepClipping, self).__init__(**kwargs)
        self.gnorm_log_ave = shared_floatx(numpy.log(initial_threshold),
                                           name='gnorm_log_ave')
        self.gnorm_log2_ave = shared_floatx(0, name='gnorm_log2_ave')
        self.adapt_steps = shared_floatx(0, name='adapt_steps')
        self.clip_threshold = shared_floatx(numpy.nan, name='clip_threshold')
        self.clip_level = shared_floatx(numpy.nan, name='clip_level')
        self.decay = decay
        self.stdevs = stdevs
        self.clip_to_mean = clip_to_mean
        self.quick_variance_convergence = quick_variance_convergence

    def compute_steps(self, previous_steps):
        # if not hasattr(self, 'threshold'):
        #    return previous_steps

        adapt_steps_up = self.adapt_steps + 1.0

        # This will quickly converge the estimate for the mean
        cut_rho_mean = tensor.minimum(self.decay,
                                      self.adapt_steps / adapt_steps_up)
        if self.quick_variance_convergence:
            cut_rho_mean2 = cut_rho_mean
        else:
            cut_rho_mean2 = self.decay

        gnorm = l2_norm(previous_steps.values())
        gnorm_log = tensor.log(l2_norm(previous_steps.values()))

        # here we quiclky converge the mean
        gnorm_log_ave_up = (cut_rho_mean * self.gnorm_log_ave +
                            (1. - cut_rho_mean) * gnorm_log)

        # this can wait as it starts from 0 anyways!
        gnorm_log2_ave_up = (cut_rho_mean2 * self.gnorm_log2_ave +
                             (1. - cut_rho_mean2) * (gnorm_log ** 2))

        clip_threshold_up = tensor.exp(
            gnorm_log_ave_up +
            tensor.sqrt(tensor.maximum(0.0,
                                       gnorm_log2_ave_up -
                                       gnorm_log_ave_up ** 2)
                        ) * self.stdevs)

        if self.clip_to_mean:
            clip_level_up = tensor.exp(gnorm_log_ave_up)
        else:
            clip_level_up = clip_threshold_up

        multiplier = tensor.switch(gnorm < clip_threshold_up,
                                   1, clip_level_up / gnorm)
        steps = OrderedDict(
            (parameter, step * multiplier)
            for parameter, step in previous_steps.items())

        return steps, [(self.adapt_steps, adapt_steps_up),
                       (self.gnorm_log_ave, gnorm_log_ave_up),
                       (self.gnorm_log2_ave, gnorm_log2_ave_up),
                       (self.clip_threshold, clip_threshold_up),
                       (self.clip_level, clip_level_up)]


logger = logging.getLogger(__name__)
class Adasecant(StepRule):
    """
    Adasecant:
        Based on the paper:
            Gulcehre, Caglar, and Yoshua Bengio.
            "ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient."
            arXiv preprint arXiv:1412.7419 (2014).
    There are some small changes in this code.
    Parameters
    ----------

    gamma_clip : float, optional
        The clipping threshold for the gamma. In general 1.8 seems to
        work fine for several tasks.
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper. Decay 0.95 seems to work fine for several tasks.
    start_var_reduction: float, optional,
        How many updates later should the variance reduction start from?
    delta_clip: float, optional,
        The threshold to clip the deltas after.
    grad_clip: float, optional,
        Apply gradient clipping for RNNs (not necessary for feedforward networks). But this is
        a constraint on the norm of the gradient per layer.
        Based on:
            Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training
            recurrent neural networks." arXiv preprint arXiv:1211.5063 (2012).
    use_adagrad: bool, optional
        Either to use clipped adagrad or not.
    use_corrected_grad: bool, optional
        Either to use correction for gradients (referred as variance
        reduction in the workshop paper).
    """
    def __init__(self, decay=0.95,
                 gamma_clip=0.0,
                 grad_clip=None,
                 start_var_reduction=0,
                 delta_clip=25,
                 gamma_reg=1e-6,
                 slow_decay=0.995,
                 use_adagrad=True,
                 perform_update=True,
                 skip_nan_inf=False,
                 use_corrected_grad=True):

        assert decay >= 0.
        assert decay < 1.

        self.start_var_reduction = start_var_reduction
        self.delta_clip = delta_clip
        self.gamma_clip = gamma_clip
        self.grad_clip = grad_clip
        self.slow_decay = slow_decay
        self.decay = shared_floatx(decay, "decay")
        self.use_corrected_grad = use_corrected_grad
        self.use_adagrad = use_adagrad
        self.gamma_reg = gamma_reg
        self.damping = 1e-7
        self.perform_update = perform_update

        # We have to bound the tau to prevent it to
        # grow to an arbitrarily large number, oftenwise
        # that causes numerical instabilities for very deep
        # networks. Note that once tau become very large, it will keep,
        # increasing indefinitely.
        self.skip_nan_inf = skip_nan_inf
        self.upper_bound_tau = 1e7
        self.lower_bound_tau = 1.5

    def compute_steps(self, grads):
        """
        .. todo::
            WRITEME
        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient. Learning rate is not being used but, pylearn2 requires a
            learning rate to be defined.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        next_step = OrderedDict()

        updates = OrderedDict({})
        eps = self.damping
        step = shared_floatx(0., name="step")

        if self.skip_nan_inf:
            #If norm of the gradients of a parameter is inf or nan don't update that parameter
            #That might be useful for RNNs.
            grads = OrderedDict({p: tensor.switch(tensor.or_(tensor.isinf(grads[p]),
                                                   tensor.isnan(grads[p])), 0, grads[p]) for
                                 p in grads.keys()})

        #Block-normalize gradients:
        nparams = len(grads.keys())

        #Apply the gradient clipping, this is only sometimes
        #necessary for RNNs and sometimes for very deep networks
        if self.grad_clip:
            assert self.grad_clip > 0.
            assert self.grad_clip <= 1., "Norm of the gradients per layer can not be larger than 1."

            gnorm = sum([g.norm(2) for g in grads.values()])
            notfinite = tensor.or_(tensor.isnan(gnorm), tensor.isinf(gnorm))

            for p, g in grads.iteritems():
                tmpg = tensor.switch(gnorm / nparams > self.grad_clip,
                                g * self.grad_clip * nparams / gnorm , g)
                grads[p] = tensor.switch(notfinite, 0.1*p, tmpg)

        tot_norm_up = 0
        tot_param_norm = 0

        fix_decay = self.slow_decay**(step + 1)
        for param in grads.keys():
            grads[param].name = "grad_%s" % param.name
            mean_grad = shared_floatx(param.get_value() * 0. + eps, name="mean_grad_%s" % param.name)
            mean_corrected_grad = shared_floatx(param.get_value() * 0 + eps, name="mean_corrected_grad_%s" % param.name)
            gnorm_sqr = shared_floatx(0.0 + eps, name="gnorm_%s" % param.name)

            prod_taus = shared_floatx((numpy.ones_like(param.get_value()) - 2*eps),
                                name="prod_taus_x_t_" + param.name)
            slow_constant = 2.1

            if self.use_adagrad:
                # sum_square_grad := \sum_i g_i^2
                sum_square_grad = shared_floatx(param.get_value(borrow=True) * 0.,
                                          name="sum_square_grad_%s" % param.name)

            """
               Initialization of accumulators
            """
            taus_x_t = shared_floatx((numpy.ones_like(param.get_value()) + eps) * slow_constant,
                               name="taus_x_t_" + param.name)
            self.taus_x_t = taus_x_t

            #Variance reduction parameters
            #Numerator of the gamma:
            gamma_nume_sqr = shared_floatx(numpy.zeros_like(param.get_value()) + eps,
                                     name="gamma_nume_sqr_" + param.name)

            #Denominator of the gamma:
            gamma_deno_sqr = shared_floatx(numpy.zeros_like(param.get_value()) + eps,
                                     name="gamma_deno_sqr_" + param.name)

            #For the covariance parameter := E[\gamma \alpha]_{t-1}
            cov_num_t = shared_floatx(numpy.zeros_like(param.get_value()) + eps,
                                name="cov_num_t_" + param.name)

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = shared_floatx(numpy.zeros_like(param.get_value()) + eps,
                                       name="msg_" + param.name)

            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = shared_floatx(param.get_value() * 0., name="msd_" + param.name)

            if self.use_corrected_grad:
                old_grad = shared_floatx(param.get_value() * 0. + eps)

            #The uncorrected gradient of previous of the previous update:
            old_plain_grad = shared_floatx(param.get_value() * 0. + eps)
            mean_curvature = shared_floatx(param.get_value() * 0. + eps)
            mean_curvature_sqr = shared_floatx(param.get_value() * 0. + eps)

            # Initialize the E[\Delta]_{t-1}
            mean_dx = shared_floatx(param.get_value() * 0.)

            # Block-wise normalize the gradient:
            norm_grad = grads[param]

            #For the first time-step, assume that delta_x_t := norm_grad
            gnorm = tensor.sqr(norm_grad).sum()

            cond = tensor.eq(step, 0)
            gnorm_sqr_o = cond * gnorm + (1 - cond) * gnorm_sqr
            gnorm_sqr_b = gnorm_sqr_o / (1 - fix_decay)

            norm_grad = norm_grad / (tensor.sqrt(gnorm_sqr_b) + eps)
            msdx = cond * norm_grad**2 + (1 - cond) * mean_square_dx
            mdx = cond * norm_grad + (1 - cond) * mean_dx

            new_prod_taus = (
                prod_taus * (1 - 1 / taus_x_t)
            )

            """
                Compute the new updated values.
            """
            # E[g_i^2]_t
            new_mean_squared_grad = (
                mean_square_grad * (1 - 1 / taus_x_t)  +
                tensor.sqr(norm_grad) / (taus_x_t)
            )
            new_mean_squared_grad.name = "msg_" + param.name

            # E[g_i]_t
            new_mean_grad = (
                mean_grad * (1 - 1 / taus_x_t) +
                norm_grad / taus_x_t
            )

            new_mean_grad.name = "nmg_" + param.name
            mg = new_mean_grad / (1 - new_prod_taus)
            mgsq = new_mean_squared_grad / (1 - new_prod_taus)

            new_gnorm_sqr = (
                gnorm_sqr_o * self.slow_decay +
                tensor.sqr(norm_grad).sum() * (1 - self.slow_decay)
            )

            # Keep the rms for numerator and denominator of gamma.
            new_gamma_nume_sqr = (
                gamma_nume_sqr * (1 - 1 / taus_x_t) +
                tensor.sqr((norm_grad - old_grad) * (old_grad - mg)) / taus_x_t
            )
            new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name

            new_gamma_deno_sqr = (
                gamma_deno_sqr * (1 - 1 / taus_x_t) +
                tensor.sqr((mg - norm_grad) * (old_grad - mg)) / taus_x_t
            )

            new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

            gamma = tensor.sqrt(gamma_nume_sqr) / (tensor.sqrt(gamma_deno_sqr + eps) + \
                                              self.gamma_reg)

            gamma.name = "gamma_" + param.name

            if self.gamma_clip and self.gamma_clip > -1:
                gamma = tensor.minimum(gamma, self.gamma_clip)

            momentum_step = gamma * mg
            corrected_grad_cand = (norm_grad + momentum_step) / (1 + gamma)

            #For starting the variance reduction.
            if self.start_var_reduction > -1:
                cond = tensor.le(self.start_var_reduction, step)
                corrected_grad = cond * corrected_grad_cand + (1 - cond) * norm_grad
            else:
                corrected_grad = norm_grad

            if self.use_adagrad:
                g = corrected_grad
                # Accumulate gradient
                new_sum_squared_grad = (
                    sum_square_grad + tensor.sqr(g)
                )
                rms_g_t = tensor.sqrt(new_sum_squared_grad)
                rms_g_t = tensor.maximum(rms_g_t, 1.0)

            #Use the gradients from the previous update
            #to compute the \nabla f(x_t) - \nabla f(x_{t-1})
            cur_curvature = norm_grad - old_plain_grad
            #cur_curvature = theano.printing.Print("Curvature: ")(cur_curvature)
            cur_curvature_sqr = tensor.sqr(cur_curvature)

            new_curvature_ave = (
                mean_curvature * (1 - 1 / taus_x_t) +
                (cur_curvature / taus_x_t)
            )
            new_curvature_ave.name = "ncurve_ave_" + param.name

            #Average average curvature
            nc_ave = new_curvature_ave / (1 - new_prod_taus)

            new_curvature_sqr_ave = (
                mean_curvature_sqr * (1 - 1 / taus_x_t) +
                (cur_curvature_sqr / taus_x_t)
            )
            new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

            #Unbiased average squared curvature
            nc_sq_ave = new_curvature_sqr_ave / (1 - new_prod_taus)

            epsilon = 1e-7
            #lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = tensor.sqrt(msdx + epsilon)

            rms_curve_t = tensor.sqrt(new_curvature_sqr_ave + epsilon)

            #This is where the update step is being defined
            delta_x_t = -(rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
            delta_x_t.name = "delta_x_t_" + param.name

            # This part seems to be necessary for only RNNs
            # For feedforward networks this does not seem to be important.
            if self.delta_clip:
                logger.info("Clipping will be applied on the adaptive step size.")
                delta_x_t = delta_x_t.clip(-self.delta_clip, self.delta_clip)
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    logger.info("Clipped adagrad is disabled.")
                    delta_x_t = delta_x_t * corrected_grad
            else:
                logger.info("Clipping will not be applied on the adaptive step size.")
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    logger.info("Clipped adagrad will not be used.")
                    delta_x_t = delta_x_t * corrected_grad

            new_taus_t = (1 - tensor.sqr(mdx) / (msdx + eps)) * taus_x_t + shared_floatx(1 + eps, "stabilized")

            #To compute the E[\Delta^2]_t
            new_mean_square_dx = (
                msdx * (1 - 1 / taus_x_t) +
                (tensor.sqr(delta_x_t) / taus_x_t)
            )

            #To compute the E[\Delta]_t
            new_mean_dx = (
                mdx * (1 - 1 / taus_x_t) +
                (delta_x_t / (taus_x_t))
            )

            #Perform the outlier detection:
            #This outlier detection is slightly different:
            new_taus_t = tensor.switch(tensor.or_(abs(norm_grad - mg) > (2 * tensor.sqrt(mgsq  - mg**2)),
                                        abs(cur_curvature - nc_ave) > (2 * tensor.sqrt(nc_sq_ave - nc_ave**2))),
                                  tensor.switch(new_taus_t > 2.5, shared_floatx(2.5), new_taus_t + shared_floatx(1.0) + eps), new_taus_t)

            #Apply the bound constraints on tau:
            new_taus_t = tensor.maximum(self.lower_bound_tau, new_taus_t)
            new_taus_t = tensor.minimum(self.upper_bound_tau, new_taus_t)

            new_cov_num_t = (
                cov_num_t * (1 - 1 / taus_x_t) +
                (delta_x_t * cur_curvature) * (1 / taus_x_t)
            )

            update_step = delta_x_t

            tot_norm_up += update_step.norm(2)
            tot_param_norm += param.norm(2)

            # Apply updates
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[mean_dx] = new_mean_dx
            updates[gnorm_sqr] = new_gnorm_sqr
            updates[gamma_nume_sqr] = new_gamma_nume_sqr
            updates[gamma_deno_sqr] = new_gamma_deno_sqr
            updates[taus_x_t] = new_taus_t
            updates[cov_num_t] = new_cov_num_t
            updates[mean_grad] = new_mean_grad
            updates[old_plain_grad] = norm_grad
            updates[mean_curvature] = new_curvature_ave
            updates[mean_curvature_sqr] = new_curvature_sqr_ave

            if self.perform_update:
                next_step[param] = -update_step

            updates[step] = step + 1
            updates[prod_taus] = new_prod_taus

            if self.use_adagrad:
                updates[sum_square_grad] = new_sum_squared_grad

            if self.use_corrected_grad:
                updates[old_grad] = corrected_grad

        return next_step, list(updates.items())

