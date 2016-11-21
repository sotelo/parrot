"""Sampling code for the parrot.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model

from datasets import parrot_stream
from model import Parrot
from utils import sample_parse
from generate import generate_wav

from io_funcs.binary_io import BinaryIOCollection
from frontend.mean_variance_norm import MeanVarianceNorm

logging.basicConfig()

data_dir = os.environ['FUEL_DATA_PATH']
args = sample_parse()

with open(os.path.join(
        args.save_dir, 'config',
        args.experiment_name + '.pkl')) as f:
    saved_args = cPickle.load(f)

assert saved_args.dataset == args.dataset

with open(os.path.join(
        args.save_dir, "pkl",
        "best_" + args.experiment_name + ".tar"), 'rb') as src:
    parameters = load_parameters(src)

if args.new_sentences:
    numpy.random.seed(1)
    spk_tr = numpy.random.random_integers(
        1, saved_args.num_speakers - 1, (args.num_samples, 1))
    spk_tr = numpy.int8(spk_tr)

    new_sentences_file = os.path.join(
        data_dir, args.dataset,
        'new_sentences.npy')

    print 'Loading sentences from : ' + new_sentences_file

    labels_tr = numpy.load(new_sentences_file)
    lengths_tr = [len(x) for x in labels_tr]
    max_length = max(lengths_tr)
    features_mask_tr = numpy.zeros(
        (args.num_samples, max_length), dtype='float32')
    padded_labels_tr = numpy.zeros(
        (args.num_samples, max_length, saved_args.input_dim), dtype='float32')

    for i, sample in enumerate(labels_tr):
        padded_labels_tr[i, :len(sample)] = sample
        features_mask_tr[i, :len(sample)] = 1.

    labels_tr = padded_labels_tr

    features_mask_tr = features_mask_tr.swapaxes(0, 1)
    labels_tr = labels_tr.swapaxes(0, 1)
else:

    test_stream = parrot_stream(
        args.dataset, saved_args.use_speaker, ('test',), args.num_samples,
        args.num_steps, sorting_mult=1)

    if saved_args.use_speaker:
        features_tr, features_mask_tr, labels_tr, spk_tr, start_flag_tr = \
            next(test_stream.get_epoch_iterator())
    else:
        features_tr, features_mask_tr, labels_tr, start_flag_tr = \
            next(test_stream.get_epoch_iterator())
        spk_tr = None

if args.speaker_id and saved_args.use_speaker:
    spk_tr = spk_tr * 0 + args.speaker_id

if args.mix and saved_args.use_speaker:
    spk_tr = spk_tr * 0
    parameters['/parrot/lookuptable.W'][0] = \
        args.mix * parameters['/parrot/lookuptable.W'][10] + \
        (1 - args.mix) * parameters['/parrot/lookuptable.W'][11]

# Set default values for old config files.
if not hasattr(saved_args, 'weak_feedback'):
    saved_args.weak_feedback = False
if not hasattr(saved_args, 'full_feedback'):
    saved_args.full_feedback = False

parrot_args = {
    'input_dim': saved_args.input_dim,
    'output_dim': saved_args.output_dim,
    'rnn_h_dim': saved_args.rnn_h_dim,
    'readouts_dim': saved_args.readouts_dim,
    'weak_feedback': saved_args.weak_feedback,
    'full_feedback': saved_args.full_feedback,
    'feedback_noise_level': None,
    'layer_normalization': saved_args.layer_normalization,
    'use_speaker': saved_args.use_speaker,
    'num_speakers': saved_args.num_speakers,
    'speaker_dim': saved_args.speaker_dim,
    'which_cost': saved_args.which_cost,
    'sampling_bias': args.sampling_bias,
    'name': 'parrot'}

parrot = Parrot(**parrot_args)

features, features_mask, labels, speaker, start_flag = \
    parrot.symbolic_input_variables()

cost, extra_updates = parrot.compute_cost(
    features, features_mask, labels, speaker, start_flag, args.num_samples)

model = Model(cost)
model.set_parameter_values(parameters)

print "Successfully loaded the parameters."

x_sample = parrot.sample_model(
    labels_tr, features_mask_tr, spk_tr, args.num_samples)

print "Successfully sampled the parrot."

x_sample = x_sample[0].swapaxes(0, 1)

io_funcs = BinaryIOCollection()

gen_file_list = []
for i, this_sample in enumerate(x_sample):
    this_sample = this_sample[:int(features_mask_tr.sum(axis=0)[i])]
    file_name = os.path.join(
        args.save_dir, 'samples',
        args.samples_name + '_' + str(i) + ".cmp")
    io_funcs.array_to_binary_file(this_sample, file_name)
    gen_file_list.append(file_name)

print "End of sampling."

norm_info_file = os.path.join(
    data_dir, args.dataset,
    'norm_info_mgc_lf0_vuv_bap_63_MVN.dat')

fid = open(norm_info_file, 'rb')
cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
fid.close()
cmp_min_max = cmp_min_max.reshape((2, -1))
cmp_min_vector = cmp_min_max[0, ]
cmp_max_vector = cmp_min_max[1, ]

denormaliser = MeanVarianceNorm(
    feature_dimension=saved_args.output_dim)
denormaliser.feature_denormalisation(
    gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

# This code was adapted from Merlin. I should add the license.

out_dimension_dict = {'bap': 1, 'lf0': 1, 'mgc': 60, 'vuv': 1}
stream_start_index = {}
file_extension_dict = {
    'mgc': '.mgc', 'bap': '.bap', 'lf0': '.lf0',
    'dur': '.dur', 'cmp': '.cmp'}
gen_wav_features = ['mgc', 'lf0', 'bap']

dimension_index = 0
for feature_name in out_dimension_dict.keys():
    stream_start_index[feature_name] = dimension_index
    dimension_index += out_dimension_dict[feature_name]

findex = 0
flen = len(gen_file_list)
for file_name in gen_file_list:
    findex = findex + 1
    dir_name = os.path.dirname(file_name)
    file_id = os.path.splitext(os.path.basename(file_name))[0]
    features, frame_number = io_funcs.load_binary_file_frame(file_name, 63)

    for feature_name in gen_wav_features:

        current_features = features[
            :, stream_start_index[feature_name]:
            stream_start_index[feature_name] +
            out_dimension_dict[feature_name]]

        gen_features = current_features

        if feature_name in ['lf0', 'F0']:
            if 'vuv' in stream_start_index.keys():
                vuv_feature = features[
                    :, stream_start_index['vuv']:stream_start_index['vuv'] + 1]

                for i in xrange(frame_number):
                    if vuv_feature[i, 0] < 0.5:
                        gen_features[i, 0] = -1.0e+10  # self.inf_float

        new_file_name = os.path.join(
            dir_name, file_id + file_extension_dict[feature_name])

        io_funcs.array_to_binary_file(gen_features, new_file_name)

generate_wav(
    os.path.join(args.save_dir, 'samples'),
    [args.samples_name + '_' + str(i) for i in range(args.num_samples)],
    sptk_dir=args.sptk_dir, world_dir=args.world_dir)
