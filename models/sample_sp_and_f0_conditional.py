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
from play.utils.mgc import mgcf02wav
from scipy.io import wavfile
from datasets.blizzard import mean_spectrum, mean_f0, std_spectrum, std_f0

from blocks.bricks.parallel import Fork

###################
# Define parameters of the model
###################

batch_size = 64 #for tpbtt
frame_size = 257 + 2
seq_size = 128
k = 100
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

experiment_name = "baseline_sp_no_fb_conditional"
num_sample = "03"

n_steps = 2048

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

phrase = "hello my friend" + "       x"

all_chars = ([chr(ord('A') + i) for i in range(26)] + [' ', '<UNK>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

phrase = [char2code[char_.upper()] for char_ in phrase]
phrase = numpy.array(phrase, dtype = 'int32').reshape([-1,1])
phrase = numpy.repeat(phrase, batch_size, axis = 1)

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

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

# dims_theta = [hidden_size_recurrent, hidden_size_mlp_theta]
# mlp_theta = MLP(activations = [Identity()], dims = dims_theta)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

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
initial_x = emitter.initial_outputs(batch_size)


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

from blocks.serialization import load_parameters
with open(save_dir+"pkl/best_"+experiment_name+".tar", 'rb') as src:
  parameters = load_parameters(src)

import logging
logging.basicConfig()

for k in parameters.keys():
  if '/emitter/mlp/' in k:
    v = parameters.pop(k)
    parameters[k.replace('/emitter/mlp/',
                '/emitter/gmmmlp/mlp/') ] = v


model.set_parameter_values(parameters)

emitter.theano_rng

def sample_step(x_tm1, h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1, ctx):

  xinp_h1_t, xgat_h1_t = inp_to_h1.apply(x_tm1)
  xinp_h2_t, xgat_h2_t = inp_to_h2.apply(x_tm1)
  xinp_h3_t, xgat_h3_t = inp_to_h3.apply(x_tm1)

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

  readout_t = h1_to_readout.apply(h1_t) + \
         h2_to_readout.apply(h2_t) + \
         h3_to_readout.apply(h3_t)

  x_t = emitter.emit(readout_t)

  mu_t, sigma_t, coeff_t = emitter.gmm_emitter.components(readout_t)

  return x_t, h1_t, h2_t, h3_t, k_t, w_t, coeff_t, ss4, a_t

(sample_x, h1, h2, h3, k, w, pi, phi, pi_att), updates = theano.scan(
    fn = sample_step,
    n_steps = n_steps,
    sequences = [],
    non_sequences = [context_oh],
    outputs_info = [initial_x.eval(), initial_h1, initial_h2, initial_h3,
    initial_kappa, initial_w, None, None, None])

phrase_mask = numpy.ones(phrase.shape, dtype = 'float32')

x_sample, pi_sample, phi_sample, pi_att_sample = function([context, context_mask], [sample_x, pi, phi, pi_att], updates = updates)(phrase,phrase_mask)

pi_sample = pi_sample.swapaxes(0,1)
phi_sample = phi_sample.swapaxes(0,1)
pi_att_sample = pi_att_sample.swapaxes(0,1)[:,:,:,0]

outputs_bp = x_sample

# ipdb.set_trace()

for this_sample in range(10):
  print "Iteration: ", this_sample
  outputs = outputs_bp

  sampled_f0 = outputs[:,:,-2]
  sampled_voiced = outputs[:,:,-1]

  print sampled_voiced.mean()
  print sampled_f0.max(), sampled_f0.min()

  outputs = outputs[:,:,:-2]
  outputs = outputs*std_spectrum + mean_spectrum
  outputs = outputs.swapaxes(0,1)
  outputs = outputs[this_sample]
  print outputs.max(), outputs.min()

  sampled_f0 = sampled_f0*std_f0 + mean_f0
  sampled_f0 = sampled_f0*sampled_voiced
  sampled_f0 = sampled_f0.swapaxes(0,1)
  sampled_f0 = sampled_f0[this_sample]

  print sampled_f0.min(), sampled_f0.max()

  # f, axarr = pyplot.subplots(2, sharex=True)
  # f.set_size_inches(10,3.5)
  # axarr[0].imshow(outputs.T)
  # #axarr[0].colorbar()
  # axarr[0].invert_yaxis()
  # axarr[0].set_ylim(0,257)
  # axarr[0].set_xlim(0,2048)
  # axarr[1].plot(sampled_f0,linewidth=3)
  # axarr[0].set_adjustable('box-forced')
  # axarr[1].set_adjustable('box-forced')
  # pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+".png")
  # pyplot.close()

  sampled_f0_corrected = sampled_f0
  sampled_f0_corrected[sampled_f0_corrected<0] = 0.

  mgc_sp = outputs 
  mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
  mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')

  # mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

  # x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
  # x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
  # wavfile.write(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+ ".wav", 16000, x_synth.astype('int16'))

  #Scaling
  outputs[outputs>11.866405] = 11.866405
  outputs[outputs<-2.0992377] = -2.0992377

  f, axarr = pyplot.subplots(5, sharex=True)
  f.set_size_inches(10,8.5)
  axarr[0].imshow(outputs.T)
  #axarr[0].colorbar()
  axarr[0].invert_yaxis()
  axarr[0].set_ylim(0, 257)
  axarr[0].set_xlim(0, 2048)
  axarr[1].plot(sampled_f0,linewidth=3)
  axarr[0].set_adjustable('box-forced')
  axarr[1].set_adjustable('box-forced')

  axarr[2].imshow(pi_sample[this_sample].T, origin = 'lower', aspect = 'auto', interpolation = 'nearest')
  axarr[3].imshow(phi_sample[this_sample].T, origin = 'lower', aspect = 'auto', interpolation = 'nearest')
  axarr[4].imshow(pi_att_sample[this_sample].T, origin = 'lower', aspect = 'auto', interpolation = 'nearest')

  pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+"_scaled.png")
  pyplot.close()

  mgc_sp = outputs 
  mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
  mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')
  mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)
  x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
  x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
  wavfile.write(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+ "_scaled.wav", 16000, x_synth.astype('int16'))




