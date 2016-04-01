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
from blocks.bricks import (Tanh, MLP, Linear,
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

from blocks.bricks.parallel import Fork

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
rec_h_dim = 2000
readouts_dim = 2000

depth_recurrent = 3
lr = 2e-4
#lr = shared_floatx(lr, "learning_rate")

att_size = 10
num_letters = 28

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "baseline_sp_simplest_conditional_2"

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
# Define model components
#################

w_init = IsotropicGaussian(0.01)
b_init = Constant(0.)

cell1 = GatedRecurrent(dim = rec_h_dim, weights_init = w_init, name = 'cell1')
cell2 = GatedRecurrent(dim = rec_h_dim, weights_init = w_init, name = 'cell2')
cell3 = GatedRecurrent(dim = rec_h_dim, weights_init = w_init, name = 'cell3')

inp_to_h1 = Fork(output_names = ['cell1_inputs', 'cell1_gates'],
        input_dim = frame_size,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'inp_to_h1')

inp_to_h2 = Fork(output_names = ['cell2_inputs', 'cell2_gates'],
        input_dim = frame_size,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'inp_to_h2')

inp_to_h3 = Fork(output_names = ['cell3_inputs', 'cell3_gates'],
        input_dim = frame_size,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'inp_to_h3')

h1_to_h2 = Fork(output_names = ['cell2_inputs', 'cell2_gates'],
        input_dim = rec_h_dim,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'h1_to_h2')

h1_to_h3 = Fork(output_names = ['cell3_inputs', 'cell3_gates'],
        input_dim = rec_h_dim,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'h1_to_h3')

h2_to_h3 = Fork(output_names = ['cell3_inputs', 'cell3_gates'],
        input_dim = rec_h_dim,
        output_dims = [rec_h_dim, 2*rec_h_dim],
        weights_init = w_init,
        biases_init = b_init,
        name = 'h2_to_h3')

h1_to_readout = Linear(input_dim = rec_h_dim,
             output_dim = readouts_dim,
             weights_init = w_init,
             biases_init = b_init,
             name = 'h1_to_readout')

h2_to_readout = Linear(input_dim = rec_h_dim,
             output_dim = readouts_dim,
             weights_init = w_init,
             biases_init = b_init,
             name = 'h2_to_readout')

h3_to_readout = Linear(input_dim = rec_h_dim,
             output_dim = readouts_dim,
             weights_init = w_init,
             biases_init = b_init,
             name = 'h3_to_readout')

h1_to_att = Fork(output_names = ['alpha', 'beta', 'kappa'],
         input_dim = rec_h_dim,
         output_dims = [att_size]*3,
         weights_init = w_init,
         biases_init = b_init,
         name = 'h1_to_att')

att_to_h1 = Fork(output_names = ['cell1_inputs', 'cell1_gates'],
          input_dim = num_letters,
          output_dims = [rec_h_dim, 2*rec_h_dim],
          weights_init = w_init,
          biases_init = b_init,
          name = 'att_to_h1')

att_to_h2 = Fork(output_names = ['cell2_inputs', 'cell2_gates'],
          input_dim = num_letters,
          output_dims = [rec_h_dim, 2*rec_h_dim],
          weights_init = w_init,
          biases_init = b_init,
          name = 'att_to_h2')

att_to_h3 = Fork(output_names = ['cell3_inputs', 'cell3_gates'],
          input_dim = num_letters,
          output_dims = [rec_h_dim, 2*rec_h_dim],
          weights_init = w_init,
          biases_init = b_init,
          name = 'att_to_h3')

activations_theta = [Rectifier()]*depth_theta

# dims_theta = [hidden_size_recurrent] + \
#              [hidden_size_mlp_theta]*depth_theta

# mlp_theta = MLP( activations = activations_theta,
#              dims = dims_theta)

dims_theta = [hidden_size_recurrent, hidden_size_mlp_theta]
mlp_theta = MLP(activations = [Identity()], dims = dims_theta)

emitter = SPF0Emitter2(mlp = mlp_theta,
                      name = "emitter",
                      weights_init = w_init,
                      biases_init = b_init)

all_blocks = [cell1, cell2, cell3, inp_to_h1, inp_to_h2, inp_to_h3,
          h1_to_h2, h1_to_h3, h2_to_h3,
          h1_to_readout, h2_to_readout, h3_to_readout,
          h1_to_att, att_to_h1, att_to_h2, att_to_h3, emitter]

[block.initialize() for block in all_blocks]

initial_h1 = shared_floatx_zeros((batch_size, rec_h_dim))
initial_h2 = shared_floatx_zeros((batch_size, rec_h_dim))
initial_h3 = shared_floatx_zeros((batch_size, rec_h_dim))
initial_kappa = shared_floatx_zeros((batch_size, att_size))
initial_w = shared_floatx_zeros((batch_size, num_letters))

def one_hot(t, r=None):
    if r is None:
        r = tensor.max(t) + 1
        
    ranges = tensor.shape_padleft(tensor.arange(r), t.ndim)
    return tensor.eq(ranges, tensor.shape_padright(t, 1))

#################
# Model
#################

f0 = tensor.matrix('f0')
voiced = tensor.matrix('voiced')
start_flag = tensor.scalar('start_flag')
sp = tensor.tensor3('spectrum')
context = tensor.imatrix('transcripts')
context_mask = tensor.matrix('transcripts_mask')

context_oh = one_hot(context, num_letters) * context_mask.dimshuffle(0,1,'x')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')

data = tensor.concatenate([sp, f0s, voiceds], 2)
x = data[:-1]
target = data[1:]

xinp_h1, xgat_h1 = inp_to_h1.apply(x)
xinp_h2, xgat_h2 = inp_to_h2.apply(x)
xinp_h3, xgat_h3 = inp_to_h3.apply(x)

u = tensor.arange(context.shape[0]).dimshuffle('x','x',0)
u = tensor.cast(u, 'float32')

def step(xinp_h1_t, xgat_h1_t, xinp_h2_t, xgat_h2_t, xinp_h3_t, xgat_h3_t,
     h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1, ctx):
  
  attinp_h1, attgat_h1 = att_to_h1.apply(w_tm1)

  h1_t = cell1.apply(xinp_h1_t + attinp_h1,
             xgat_h1_t + attgat_h1, h1_tm1, iterate = False)
  h1inp_h2, h1gat_h2 = h1_to_h2.apply(h1_t)
  h1inp_h3, h1gat_h3 = h1_to_h3.apply(h1_t)

  a_t, b_t, k_t = h1_to_att.apply(h1_t)

  a_t = tensor.exp(a_t)
  b_t = tensor.exp(b_t)
  k_t = k_tm1 + tensor.exp(k_t)

  a_t = a_t.dimshuffle(0,1,'x')
  b_t = b_t.dimshuffle(0,1,'x')
  k_t_ = k_t.dimshuffle(0,1,'x')

  #batch size X att size X len context
  ss1 = (k_t_-u)**2
  ss2 = -b_t*ss1
  ss3 = a_t*tensor.exp(ss2)
  ss4 = ss3.sum(axis = 1)

  # batch size X len context X num letters

  ss5 = ss4.dimshuffle(0,1,'x')
  ss6 = ss5*ctx.dimshuffle(1,0,2)
  w_t  = ss6.sum(axis = 1)

  # batch size X num letters
  attinp_h2, attgat_h2 = att_to_h2.apply(w_t)
  attinp_h3, attgat_h3 = att_to_h3.apply(w_t)

  h2_t = cell2.apply(xinp_h2_t + h1inp_h2 + attinp_h2,
             xgat_h2_t + h1gat_h2 + attgat_h2, h2_tm1,
             iterate = False)

  h2inp_h3, h2gat_h3 = h2_to_h3.apply(h2_t)

  h3_t = cell3.apply(xinp_h3_t + h1inp_h3 + h2inp_h3 + attinp_h3,
             xgat_h3_t + h1gat_h3 + h2gat_h3 + attgat_h3, h3_tm1,
             iterate = False)

  return h1_t, h2_t, h3_t, k_t, w_t

(h1, h2, h3, kappa, w), updates = theano.scan(
  fn = step,
  sequences = [xinp_h1, xgat_h1, xinp_h2, xgat_h2, xinp_h3, xgat_h3],
  non_sequences = [context_oh],
  outputs_info = [initial_h1, initial_h2, initial_h3, initial_kappa, initial_w])

readouts = h1_to_readout.apply(h1) + h2_to_readout.apply(h2) + h3_to_readout.apply(h3)

cost = emitter.cost(readouts, target)
cost = cost.mean() + 0.*start_flag
cost.name = 'nll'

cg = ComputationGraph(cost)
model = Model(cost)
parameters = cg.parameters

extra_updates = []
extra_updates.append((initial_h1, tensor.switch(start_flag, 0.*initial_h1, h1[-1])))
extra_updates.append((initial_h2, tensor.switch(start_flag, 0.*initial_h2, h2[-1])))
extra_updates.append((initial_h3, tensor.switch(start_flag, 0.*initial_h3, h3[-1])))
extra_updates.append((initial_kappa, tensor.switch(start_flag, 0.*initial_kappa, kappa[-1])))
extra_updates.append((initial_w, tensor.switch(start_flag, 0.*initial_w, w[-1])))

states = [initial_h1, initial_h2, initial_h3, initial_kappa, initial_w]

#################
# Monitoring vars
#################

monitoring_variables = [cost]

#################
# Algorithm
#################

n_batches = 10
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
      states = states,
      every_n_batches = n_batches,
      before_first_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()