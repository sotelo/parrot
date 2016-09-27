"""Sampling code for the parrot.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model
from blizzard import blizzard_stream

from utils import (
    char2code, mean_f0, mean_mgc, mean_spectrum, plot_sample, sample_parse,
    std_f0, spectrum_to_audio, std_mgc, std_spectrum)

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

test_stream = blizzard_stream(
    ('test',), args.num_samples, args.num_steps - 1)

if saved_args.model == "simple":
    from model import SimpleParrot as Parrot
    parrot_args = {
        'num_freq': saved_args.num_freq,
        'k': saved_args.num_mixture,
        'k_f0': saved_args.k_f0,
        'rnn1_h_dim': saved_args.rnn1_size,
        'att_size': saved_args.size_attention,
        'num_letters': saved_args.num_letters,
        'sampling_bias': args.sampling_bias,
        'attention_type': saved_args.attention_type,
        'attention_alignment': saved_args.attention_alignment,
        'name': 'parrot'}
    parrot = Parrot(**parrot_args)
else:
    from model import Parrot
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
    start_flag, args.num_samples, args.num_steps)

model = Model(cost)
model.set_parameter_values(parameters)

phrase = args.phrase + "  "
phrase = [char2code[char_.upper()] for char_ in phrase]
phrase = numpy.array(phrase, dtype='int32').reshape([-1, 1])
phrase = numpy.repeat(phrase, args.num_samples, axis=1).T
phrase_mask = numpy.ones(phrase.shape, dtype='float32')

[x_sample, sampled_pi, sampled_phi, sampled_pi_att] = parrot.sample_model(
    phrase, phrase_mask, args.num_samples,
    args.num_steps, saved_args.use_spectrum)

sampled_pi_att = sampled_pi_att[:, :, :, 0]

sampled_spectrum = x_sample[:, :, :-2].swapaxes(0, 1)
sampled_f0 = x_sample[:, :, -2].swapaxes(0, 1)
sampled_voiced = x_sample[:, :, -1].swapaxes(0, 1)
sampled_pi = sampled_pi.swapaxes(0, 1)
sampled_phi = sampled_phi.swapaxes(0, 1)
sampled_pi_att = sampled_pi_att.swapaxes(0, 1)

if saved_args.use_spectrum:
    mean_data = mean_spectrum
    std_data = std_spectrum
else:
    mean_data = mean_mgc
    std_data = std_mgc

sampled_spectrum = sampled_spectrum * std_data + mean_data
sampled_f0 = sampled_f0 * std_f0 + mean_f0
sampled_f0 = sampled_f0 * sampled_voiced

for this_sample in range(args.num_samples):
    print "Iteration: ", this_sample

    spectrum = sampled_spectrum[this_sample]
    f0 = sampled_f0[this_sample]
    pi = sampled_pi[this_sample]
    phi = sampled_phi[this_sample]
    pi_att = sampled_pi_att[this_sample]

    plot_sample(
        spectrum, f0, pi, phi, pi_att,
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + '_' + str(this_sample) + ".png"))

    spectrum_to_audio(
        spectrum, f0, saved_args.use_spectrum,
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + '_' + str(this_sample) + ".wav"))

print "End of sampling."
