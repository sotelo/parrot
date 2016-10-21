"""Sampling code for the parrot.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model

from blizzard import aligned_stream
from model import NewPhonemesParrot
from utils import sample_parse

from io_funcs.binary_io import BinaryIOCollection

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

test_stream = aligned_stream(
    ('test',), args.num_samples, args.num_steps, sorting_mult=1)

parrot_args = {
    'input_dim': saved_args.input_dim,
    'output_dim': saved_args.output_dim,
    'rnn_h_dim': saved_args.rnn_h_dim,
    'readouts_dim': saved_args.readouts_dim,
    'name': 'parrot'}

parrot = NewPhonemesParrot(**parrot_args)

features, features_mask, labels, start_flag = \
    parrot.symbolic_input_variables()

cost, extra_updates = parrot.compute_cost(
    features, features_mask, labels, start_flag, args.num_samples)

model = Model(cost)
model.set_parameter_values(parameters)

features_tr, features_mask_tr, labels_tr, start_flag_tr = \
    next(test_stream.get_epoch_iterator())

x_sample = parrot.sample_model(
    labels_tr, features_mask_tr, args.num_samples)

x_sample = x_sample[0].swapaxes(0, 1)

io_fun = BinaryIOCollection()

gen_file_list = []
for i, this_sample in enumerate(x_sample):
    this_sample = this_sample[:features_mask_tr.sum(axis=0)[i]]
    file_name = os.path.join(
        args.save_dir, 'samples',
        args.samples_name + '_' + str(i) + ".cmp")
    io_fun.array_to_binary_file(this_sample, file_name)
    gen_file_list.append(file_name)

print "End of sampling."

import ipdb; ipdb.set_trace()

# importing from merlin
import configuration
cfg=configuration.cfg
config_file='/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/conf/acoustic_slt_arctic_full.conf'
cfg.configure(config_file, use_logging=False)

from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

norm_info_file = '/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/norm_info_mgc_lf0_vuv_bap_187_MVN.dat'
fid = open(norm_info_file, 'rb')
cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
fid.close()
cmp_min_max = cmp_min_max.reshape((2, -1))
cmp_min_vector = cmp_min_max[0, ]
cmp_max_vector = cmp_min_max[1, ]

assert saved_args.output_dim == 187
denormaliser = MeanVarianceNorm(feature_dimension=saved_args.output_dim)
denormaliser.feature_denormalisation(
    gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

var_file_dict = {
    'mgc': '/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/var/mgc_180',
    'vuv': '/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/var/vuv_1',
    'lf0': '/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/var/lf0_3',
    'bap': '/Tmp/sotelo/code/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/var/bap_3'}

generator = ParameterGeneration(
    gen_wav_features=['mgc', 'lf0', 'bap'], enforce_silence=False)
generator.acoustic_decomposition(
    gen_file_list,
    saved_args.output_dim,
    {'mgc': 180, 'vuv': 1, 'lf0': 3, 'bap': 3},
    {'mgc': '.mgc', 'bap': '.bap', 'lf0': '.lf0', 'cmp': '.cmp'},
    var_file_dict,
    do_MLPG=True,
    cfg=None)

from generate import generate_wav

generate_wav(
    os.path.join(args.save_dir, 'samples'),
    [args.samples_name + '_' + str(i) for i in range(args.num_samples)], cfg)
