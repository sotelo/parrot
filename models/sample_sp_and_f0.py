import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os
from play.utils.mgc import mgcf02wav
import pysptk as SPTK
from scipy.io import wavfile
from blocks.serialization import load, load_parameters
from blocks.graph import ComputationGraph
import ipdb

from fuel.transformers import (Mapping, FilterSources,
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from play.datasets.blizzard import Blizzard
from play.toy.segment_transformer import SegmentSequence
from parrot.datasets.blizzard import mean_spectrum, mean_f0, std_spectrum, std_f0

import logging
logging.basicConfig()

batch_size = 10

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

data_hidden_states = True

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "baseline_sp_simplest_gmm_f0"
num_sample = "01"

print "Sampling: " + experiment_name

with open(save_dir+"pkl/best_"+experiment_name+".tar", 'rb') as src:
	parameters = load_parameters(src)

#####################################
# Model Construction
#####################################

from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)
from blocks.bricks.recurrent import LSTM, RecurrentStack, GatedRecurrent

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                                SPF0Emitter2)

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

dims_theta = [hidden_size_recurrent, hidden_size_mlp_theta]
mlp_theta = MLP(activations = [Identity()], dims = dims_theta)

# mlp_theta = MLP( activations = activations_theta,
#              dims = dims_theta)

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

#######################################

steps = 2048
n_samples = batch_size

from blocks.utils import shared_floatx_zeros, shared_floatx

states = generator.transition.apply.outputs
states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent)) for name in states}

from theano import tensor, function
f0 = tensor.matrix('f0')
voiced = tensor.matrix('voiced')
start_flag = tensor.scalar('start_flag')
sp = tensor.tensor3('sp')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')
x = tensor.concatenate([sp, f0s, voiceds], 2)

states = generator.transition.apply.outputs

states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
        for name in states}

cost_matrix = generator.cost_matrix(x, **states)
cg = ComputationGraph(cost_matrix)

from blocks.model import Model

model = Model(cost_matrix)

k2 = [key for key in model.get_parameter_dict().keys() if key not in parameters.keys()]
k1 = [key for key in parameters.keys() if key not in model.get_parameter_dict().keys()]

#model.get_parameter_values()[k2]

parameters2 = parameters.copy()

for k in parameters2.keys():
	if '/generator/readout/emitter/mlp/' in k:
		v = parameters2.pop(k)
		parameters2[k.replace('/generator/readout/emitter/mlp/',
							  '/generator/readout/emitter/gmmmlp/mlp/') ] = v

model.set_parameter_values(parameters2)

# import ipdb
# ipdb.set_trace()

#print function([f0, sp, voiced], cost_matrix, updates = extra_updates)(x_tr[0],x_tr[1],x_tr[2])

#generator.generate(n_steps=steps, batch_size=n_samples, iterate=True, **states)

#states = {}

if data_hidden_states:

	from parrot.datasets.blizzard import blizzard_stream
	from play.utils import regex_final_value
	from blocks.filter import VariableFilter

	train_stream = blizzard_stream(('train',), batch_size, seq_length = 500)
	extra_updates = []
	for name, var in states.items():
	  update = VariableFilter(theano_name_regex=regex_final_value(name)
	                  )(cg.auxiliary_variables)[0]
	  extra_updates.append((var, update))

	from theano import function
	x_tr = next(train_stream.get_epoch_iterator())
	function([f0, sp, start_flag, voiced], [], updates = extra_updates, on_unused_input='warn')(*x_tr)

sample = ComputationGraph(
	generator.generate(n_steps=steps,
		batch_size=n_samples,
		iterate=True, **states))
sample_fn = sample.get_theano_function()

outputs_bp = sample_fn()[-2]

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

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(10,3.5)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+".png")
	pyplot.close()

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

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(10,3.5)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+"_scaled.png")
	pyplot.close()

	mgc_sp = outputs 
	mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
	mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')
	mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)
	x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
	x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
	wavfile.write(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+ "_scaled.wav", 16000, x_synth.astype('int16'))
