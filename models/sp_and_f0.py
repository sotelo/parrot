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
from blocks.utils import shared_floatx_zeros, shared_floatx

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                                SPF0Emitter)

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

depth_recurrent = 3
lr = 2e-4
#lr = shared_floatx(lr, "learning_rate")

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "baseline_sp_no_fb_new_f0"

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

emitter = SPF0Emitter(mlp = mlp_theta,
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
cost.name = "nll"

cg = ComputationGraph(cost)

model = Model(cost)

# from theano import function
# function([f0, sp, start_flag, voiced], cost)(*x_tr)

transition_matrix = VariableFilter(
            theano_name_regex="state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype=floatX))

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

mean_data = x.mean(axis = (0,1)).copy(name="data_mean")
sigma_data = x.std(axis = (0,1)).copy(name="data_std")
max_data = x.max(axis = (0,1)).copy(name="data_max")
min_data = x.min(axis = (0,1)).copy(name="data_min")

monitoring_variables = [cost]

data_monitoring = [mean_data, sigma_data,
                     max_data, min_data]

readout = generator.readout
readouts = VariableFilter( applications = [readout.readout],
    name_regex = "output")(cg.variables)[0]

mu, sigma, binary = readout.emitter.components(readouts)

min_sigma = sigma.min().copy(name="sigma_min")
mean_sigma = sigma.mean().copy(name="sigma_mean")
max_sigma = sigma.max().copy(name="sigma_max")

min_mu = mu.min().copy(name="mu_min")
mean_mu = mu.mean().copy(name="mu_mean")
max_mu = mu.max().copy(name="mu_max")

min_binary = binary.min().copy(name="binary_min")
mean_binary = binary.mean().copy(name="binary_mean")
max_binary = binary.max().copy(name="binary_max")

data_monitoring += [mean_sigma, min_sigma,
    min_mu, max_mu, mean_mu, max_sigma,
    mean_binary, min_binary, max_binary]

#################
# Algorithm
#################

n_batches = 1
n_batches_valid = 200

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
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