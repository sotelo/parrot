"""Sampling code for the parrot.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model
from datasets.blizzard import blizzard_stream
from utils import char2code, sample_parse

logging.basicConfig()

parser = sample_parse()
args = parser.parse_args()

with open(os.path.join(
        args.save_dir, 'config',
        args.experiment_name + '.pkl')) as f:
    saved_args = cPickle.load(f)

with open(os.path.join(
        args.save_dir, "pkl",
        "best_" + args.experiment_name + ".tar"), 'rb') as src:
    parameters = load_parameters(src)

parameters = {k.replace('parrot/', ''): v for k, v in parameters.items()}

if args.experiment_name == "incomplete_model":
    parameters['/data_to_rnn2/fork_rnn2_cell1_inputs.b'] = \
        numpy.zeros((saved_args.rnn2_size), dtype='float32')
    parameters['/data_to_rnn2/fork_rnn2_cell1_gates.b'] = \
        numpy.zeros((2 * saved_args.rnn2_size), dtype='float32')
    parameters['/data_to_rnn2/fork_rnn2_cell1_inputs.W'] = \
        numpy.zeros((saved_args.num_freq + 2,
                     saved_args.rnn2_size), dtype='float32')
    parameters['/data_to_rnn2/fork_rnn2_cell1_gates.W'] = \
        numpy.zeros((saved_args.num_freq + 2,
                     2 * saved_args.rnn2_size), dtype='float32')

test_stream = blizzard_stream(
    ('test',), args.num_samples, args.num_steps - 1)

epoch_iterator = test_stream.get_epoch_iterator()

while True:
    f0_tr, f0_mask_tr, spectrum_tr, transcripts_tr, \
        transcripts_mask_tr, start_flag_tr, voiced_tr = \
        next(epoch_iterator)
    if len(f0_tr) == args.num_steps:
        break

if saved_args.model == "simple":
    from models.model import SimpleParrot as Parrot
    parrot_args = {
        'num_freq': saved_args.num_freq,
        'k': saved_args.num_mixture,
        'k_f0': saved_args.k_f0,
        'rnn1_h_dim': saved_args.rnn1_size,
        'att_size': saved_args.size_attention,
        'num_letters': saved_args.num_letters,
        'sampling_bias': 0.,
        'name': 'parrot'}
    parrot = Parrot(**parrot_args)
else:
    from models.model import Parrot
    parrot = Parrot(
        num_freq=saved_args.num_freq,
        k=saved_args.num_mixture,
        k_f0=saved_args.k_f0,
        rnn1_h_dim=saved_args.rnn1_size,
        rnn2_h_dim=saved_args.rnn2_size,
        att_size=saved_args.size_attention,
        num_letters=saved_args.num_letters,
        sampling_bias=args.sampling_bias,
        name='parrot')

f0, f0_mask, voiced, spectrum, transcripts, transcripts_mask, start_flag = \
    parrot.symbolic_input_variables()

cost, extra_updates = parrot.compute_cost(
    f0, f0_mask, voiced, spectrum, transcripts, transcripts_mask,
    start_flag, saved_args.num_samples, saved_args.seq_length)

# sample_x, updates_sample = parrot.sample_model(
#     transcripts, transcripts_mask, args.num_steps, args.num_samples)

model = Model(cost)
model.set_parameter_values(parameters)

phrase = args.phrase + "  "
phrase = [char2code[char_.upper()] for char_ in phrase]
phrase = numpy.array(phrase, dtype='int32').reshape([-1, 1])
phrase = numpy.repeat(phrase, args.num_samples, axis=1).T
phrase_mask = numpy.ones(phrase.shape, dtype='float32')

if args.one_step_sampling:
    one_step = parrot.sample_one_step(args.num_samples)

    results = numpy.zeros(
        (args.num_steps, args.num_samples, parrot.num_freq + 2))

    for num_step in range(args.num_steps):
        print "Step: ", num_step
        old_x = numpy.concatenate([
            spectrum_tr[num_step],
            numpy.expand_dims(f0_tr[num_step], axis=1),
            numpy.expand_dims(voiced_tr[num_step], axis=1)], 1)

        results[num_step] = one_step(
            old_x, transcripts_tr, transcripts_mask_tr)
    x_sample = results

else:
    [x_sample, sampled_pi, sampled_phi, sampled_pi_att] = parrot.sample_model(
        phrase, phrase_mask, args.num_samples, args.num_steps)

# Clean this code!

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage
num_sample = "02"

from parrot.datasets.blizzard import (
    mean_spectrum, mean_f0, std_spectrum, std_f0)
import pysptk as SPTK
from play.utils.mgc import mgcf02wav
from scipy.io import wavfile
import numpy
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot


sampled_spectrum = x_sample[:, :, :-2].swapaxes(0, 1)
sampled_f0 = x_sample[:, :, -2].swapaxes(0, 1)
sampled_voiced = x_sample[:, :, -1].swapaxes(0, 1)
sampled_pi = sampled_pi.swapaxes(0, 1)
sampled_phi = sampled_phi.swapaxes(0, 1)
sampled_pi_att = sampled_pi_att.swapaxes(0, 1)[:, :, :, 0]

sampled_spectrum = sampled_spectrum * std_spectrum + mean_spectrum
sampled_f0 = sampled_f0 * std_f0 + mean_f0
sampled_f0 = sampled_f0 * sampled_voiced

for this_sample in range(10):
    print "Iteration: ", this_sample

    sample_spectrum = sampled_spectrum[this_sample]
    sample_f0 = sampled_f0[this_sample]

    f, axarr = pyplot.subplots(5, sharex=True)
    f.set_size_inches(10, 8.5)
    axarr[0].imshow(sample_spectrum.T)
    axarr[0].invert_yaxis()
    axarr[0].set_ylim(0, 257)
    axarr[0].set_xlim(0, 2048)
    axarr[1].plot(sample_f0, linewidth=1)
    axarr[0].set_adjustable('box-forced')
    axarr[1].set_adjustable('box-forced')

    axarr[2].imshow(sampled_pi[this_sample].T, origin='lower',
                    aspect='auto', interpolation='nearest')
    axarr[3].imshow(sampled_phi[this_sample].T, origin='lower',
                    aspect='auto', interpolation='nearest')
    axarr[4].imshow(sampled_pi_att[this_sample].T, origin='lower',
                    aspect='auto', interpolation='nearest')

    pyplot.savefig(
        args.save_dir + "samples/best_" + args.experiment_name +
        num_sample + str(this_sample) + ".png")
    pyplot.close()

    mgc_sp_test = numpy.hstack([sample_spectrum, sample_spectrum[:, ::-1][:, 1:-1]])
    mgc_sp_test = mgc_sp_test.astype('float64').copy(order='C')
    mgc_reconstruct = numpy.apply_along_axis(
        SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma,
        eps=0.0012, etype=1, itype=2)
    x_synth = mgcf02wav(mgc_reconstruct, sample_f0)
    x_synth = .95 * x_synth / max(abs(x_synth)) * 2**15
    wavfile.write(
        args.save_dir + "samples/best_" + args.experiment_name +
        num_sample + str(this_sample) + ".wav", 16000,
        x_synth.astype('int16'))
print "End"
