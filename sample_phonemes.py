"""Sampling code for the parrot.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model

from blizzard import phonemes_stream
from model import PhonemesParrot as Parrot
from utils import (
    mean_f0, mean_spectrum, plot_spectrum, sample_parse,
    std_f0, spectrum_to_audio, std_spectrum)

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

test_stream = phonemes_stream(('test',), args.num_samples)

parrot_args = {
    'num_freq': saved_args.num_freq,
    'k': saved_args.num_mixture,
    'rnn1_h_dim': saved_args.rnn1_size,
    'num_phonemes': saved_args.num_phonemes,
    'num_phonemes': saved_args.num_phonemes,
    'sampling_bias': args.sampling_bias,
    'name': 'parrot'}

parrot = Parrot(**parrot_args)

f0, voiced, spectrum, phonemes = \
    parrot.symbolic_input_variables()

cost, extra_updates = parrot.compute_cost(
    f0, voiced, spectrum, phonemes, args.num_samples)

model = Model(cost)
model.set_parameter_values(parameters)

f0_tr, phonemes_tr, spectrum_tr, voiced_tr = \
    next(test_stream.get_epoch_iterator())

x_sample = parrot.sample_model(
    f0_tr, phonemes_tr, voiced_tr, args.num_samples)

real_spectrum = spectrum_tr.swapaxes(0, 1)
sampled_spectrum = x_sample[0][:, :, :-2].swapaxes(0, 1)
sampled_f0 = f0_tr.swapaxes(0, 1)
sampled_voiced = voiced_tr.swapaxes(0, 1)

mean_data = mean_spectrum
std_data = std_spectrum

sampled_spectrum = sampled_spectrum * std_data + mean_data
real_spectrum = real_spectrum * std_data + mean_data
sampled_f0 = sampled_f0 * std_f0 + mean_f0
sampled_f0 = sampled_f0 * sampled_voiced

for this_sample in range(args.num_samples):
    print "Iteration: ", this_sample

    spectrum = sampled_spectrum[this_sample]
    f0 = sampled_f0[this_sample]

    plot_spectrum(
        real_spectrum[this_sample], spectrum, f0,
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + '_' + str(this_sample) + ".png"))

    spectrum_to_audio(
        spectrum, f0, True,
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + '_' + str(this_sample) + ".wav"))

    spectrum_to_audio(
        real_spectrum[this_sample], f0, True,
        os.path.join(
            args.save_dir, 'samples',
            'real' + '_' + str(this_sample) + ".wav"))

print "End of sampling."
