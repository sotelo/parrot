import ipdb
import numpy
import theano
import matplotlib
import os
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Scale,
                               RMSProp, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)
from blocks.bricks.recurrent import LSTM, RecurrentStack, GatedRecurrent
from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx_zeros, shared_floatx, shared_floatx_zeros, shared_floatx_zeros_matching

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                                SPF0Emitter2)

from play.extensions import Flush, LearningRateSchedule, TimedFinish
from play.extensions.plot import Plot

import pysptk as SPTK

###################
# Define parameters of the model
###################

batch_size = 64 #for tpbtt
frame_size = 257 + 2
seq_size = 128
k = 20
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 2000

depth_theta = 4
hidden_size_mlp_theta = 2000
hidden_size_recurrent = 2000

weight_noise = True

depth_recurrent = 3
lr = 2e-4
#lr = shared_floatx(lr, "learning_rate")

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "baseline_sp_no_fb_gmm_f0_wn"

load_params = "baseline_sp_no_fb_gmm_f0"

#################
# Prepare dataset
#################

from parrot.datasets.blizzard import blizzard_stream

train_stream = blizzard_stream(('train',), batch_size)
valid_stream = blizzard_stream(
                  ('valid',), batch_size, seq_length = 200,
                  num_examples = 64, sorting_mult = 1)

x_tr = next(train_stream.get_epoch_iterator())

#################
# Model
#################

f0 = tensor.matrix('f0')
voiced = tensor.matrix('voiced')
start_flag = tensor.scalar('start_flag')
sp = tensor.tensor3('spectrum')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')

x = tensor.concatenate([sp, f0s, voiceds], 2)

#x = tensor.tensor3('features')

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

#feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                   name = "gru_{}".format(i) ) for i in range(depth_recurrent)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

# dims_theta = [hidden_size_recurrent, hidden_size_mlp_theta]
# mlp_theta = MLP(activations = [Identity()], dims = dims_theta)

emitter = SPF0Emitter2(mlp = mlp_theta,
                      name = "emitter")

source_names = [name for name in transition.apply.states if 'states' in name]
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names,
    emitter=emitter,
    #feedback_brick = feedback,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.push_initialization_config()

generator.transition.biases_init = IsotropicGaussian(0.01,1)
generator.transition.push_initialization_config()

generator.initialize()

states = {}
states = generator.transition.apply.outputs

states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
        for name in states}

cost_matrix = generator.cost_matrix(x, **states)

cost = cost_matrix.mean() + 0.*start_flag

cg = ComputationGraph(cost)

model = Model(cost)

simple_cost = cost
simple_cost.name = "nll"

parameters = cg.parameters

# from theano import function
# function([f0, sp, start_flag, voiced], cost)(*x_tr)

transition_matrix = VariableFilter(
            theano_name_regex="state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype=floatX))

if load_params:
  from blocks.serialization import load_parameters
  with open(save_dir+"pkl/best_"+load_params+".tar", 'rb') as src:
    loaded_parameters = load_parameters(src)

  for k in loaded_parameters.keys():
    if '/generator/readout/emitter/mlp/' in k:
      v = loaded_parameters.pop(k)
      loaded_parameters[k.replace('/generator/readout/emitter/mlp/',
                  '/generator/readout/emitter/gmmmlp/mlp/') ] = v

  model.set_parameter_values(loaded_parameters)

if weight_noise:
  from theano.sandbox.rng_mrg import MRG_RandomStreams
  from blocks.roles import add_role, PARAMETER

  def apply_noise(computation_graph, variables, level = 0.075, seed=None):
      if not seed:
          seed = 1
      rng = MRG_RandomStreams(seed)

      replace = {}
      variances = []
      for nvar, variable in enumerate(variables):
        variance = shared_floatx_zeros_matching(variable, name = "variance_" + str(nvar))
        variance.set_value(variance.get_value() + numpy.log(level**2))
        variances.append(variance)
        replace[variable] = (variable +
            rng.normal(variable.shape)*tensor.sqrt(tensor.exp(variance) + 1e-5))
        add_role(variance, PARAMETER)

      return computation_graph.replace(replace), variances

  cost_with_noise_cg, variances = apply_noise(cg, cg.parameters, level = 0.075)
  cost = cost_with_noise_cg.outputs[0]

  prior = 0.075**2

  # for variance in variances:
  #   cost += (0.5*(tensor.exp(variance)/prior - variance)).sum()

  cost.name = 'reg_cost'
  model = Model(cost)

  parameters += variances

from play.utils import regex_final_value
extra_updates = []
for name, var in states.items():
  update = tensor.switch(start_flag, 0.*var,
              VariableFilter(theano_name_regex=regex_final_value(name)
                  )(cg.auxiliary_variables)[0])
  extra_updates.append((var, update))

#################
# Monitoring vars
#################

monitoring_variables = [simple_cost]

#################
# Algorithm
#################

n_batches = 1
n_batches_valid = 200

algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))
algorithm.add_updates(extra_updates)
lr = algorithm.step_rule.components[1].learning_rate

train_monitor = TrainingDataMonitoring(
    variables=monitoring_variables + [lr],
    every_n_batches=n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     monitoring_variables,
     valid_stream,
     every_n_batches = n_batches_valid,
     before_first_epoch = False,
     prefix="valid")

extensions=[
    Timing(every_n_batches=n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches = n_batches_valid),
    Plot(save_dir+ "progress/" +experiment_name+".png",
     [['train_nll',
       'valid_nll'], ['train_learning_rate']],
     every_n_batches=n_batches_valid,
     email=False),
    Checkpoint(
       save_dir+"pkl/best_"+experiment_name+".tar",
       #save_separately = ['parameters'],
       save_main_loop = False,
       use_cpickle=True
    ).add_condition(
       ['after_batch'], predicate=OnLogRecord('valid_nll_best_so_far')),
    Printing(every_n_batches = n_batches),
    Flush(every_n_batches=n_batches,
          before_first_epoch = True),
    LearningRateSchedule(lr,
      'valid_nll',
      states = states.values(),
      every_n_batches = n_batches,
      before_first_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()