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
from utils import (
    attention_plot, sample_parse, create_animation, numpy_one_hot)
from generate import generate_wav

logging.basicConfig()

data_dir = os.environ['FUEL_DATA_PATH']
args = sample_parse()

with open(os.path.join(
        args.save_dir, 'config',
        args.experiment_name + '.pkl')) as f:
    saved_args = cPickle.load(f)

assert saved_args.dataset == args.dataset

if args.use_last:
    params_mode = 'last_'
else:
    params_mode = 'best_'

args.samples_name = params_mode + args.samples_name

with open(os.path.join(
        args.save_dir, "pkl",
        params_mode + args.experiment_name + ".tar"), 'rb') as src:
    parameters = load_parameters(src)

test_stream = parrot_stream(
    args.dataset, saved_args.use_speaker, ('test',), args.num_samples,
    args.num_steps, sorting_mult=1, labels_type=saved_args.labels_type,
    raw_data=args.plot_raw)

data_tr = next(test_stream.get_epoch_iterator())
data_tr = {
    source: data for source, data in zip(test_stream.sources, data_tr)}

print "Loaded sources from test_stream: ", data_tr.keys()
features_tr = data_tr.get('features', None)
features_mask_tr = data_tr.get('features_mask', None)
speaker_tr = data_tr.get('speaker_index', None)
labels_tr = data_tr.get('labels', None)
labels_mask_tr = data_tr.get('labels_mask', None)
start_flag_tr = data_tr.get('start_flag', None)
raw_audio_tr = data_tr.get('raw_audio', None)

if args.random_speaker:
    numpy.random.seed(1)
    speaker_tr = numpy.random.random_integers(
        1, saved_args.num_speakers - 1, (args.num_samples, 1))
    speaker_tr = numpy.int8(speaker_tr)

if args.phrase is not None:
    import pickle
    data_path = os.environ['FUEL_DATA_PATH']
    char2code_path = os.path.join(data_path, args.dataset, 'char2code.pkl')
    with open(char2code_path, 'r') as f:
        char2code = pickle.load(f)
    labels_tr = numpy.array([char2code[x] for x in args.phrase], dtype='int8')
    labels_tr = numpy.tile(labels_tr, (args.num_samples, 1))
    labels_mask_tr = numpy.ones(labels_tr.shape, dtype='float32')

if args.speaker_id and saved_args.use_speaker:
    speaker_tr = speaker_tr * 0 + args.speaker_id

if args.mix and saved_args.use_speaker:
    speaker_tr = speaker_tr * 0
    parameters['/parrot/lookuptable.W'][0] = \
        args.mix * parameters['/parrot/lookuptable.W'][15] + \
        (1 - args.mix) * parameters['/parrot/lookuptable.W'][11]

# Set default values for old config files.
# if not hasattr(saved_args, 'weak_feedback'):
#     saved_args.weak_feedback = False
# if not hasattr(saved_args, 'full_feedback'):
#     saved_args.full_feedback = False
if not hasattr(saved_args, 'raw_output'):
    saved_args.raw_output = False

parrot_args = {
    'input_dim': saved_args.input_dim,
    'output_dim': saved_args.output_dim,
    'rnn_h_dim': saved_args.rnn_h_dim,
    'readouts_dim': saved_args.readouts_dim,
    'weak_feedback': saved_args.weak_feedback,
    'full_feedback': saved_args.full_feedback,
    'feedback_noise_level': None,
    'layer_norm': saved_args.layer_norm,
    'use_speaker': saved_args.use_speaker,
    'num_speakers': saved_args.num_speakers,
    'speaker_dim': saved_args.speaker_dim,
    'which_cost': saved_args.which_cost,
    'num_characters': saved_args.num_characters,
    'attention_type': saved_args.attention_type,
    'attention_alignment': saved_args.attention_alignment,
    'sampling_bias': args.sampling_bias,
    'sharpening_coeff': args.sharpening_coeff,
    'timing_coeff': args.timing_coeff,
    'encoder_type': saved_args.encoder_type,
    'raw_output': saved_args.raw_output,
    'name': 'parrot'}

parrot = Parrot(**parrot_args)

features, features_mask, labels, labels_mask, speaker, start_flag, raw_audio = \
    parrot.symbolic_input_variables()

cost, extra_updates, attention_vars, cost_raw = parrot.compute_cost(
    features, features_mask, labels, labels_mask,
    speaker, start_flag, args.num_samples, raw_audio=raw_audio)

model = Model(cost)
model.set_parameter_values(parameters)

print "Successfully loaded the parameters."

if args.sample_one_step:
    gen_x, gen_k, gen_w, gen_pi, gen_phi, gen_pi_att = \
        parrot.sample_using_input(data_tr, args.num_samples)
else:
    gen_x, gen_k, gen_w, gen_pi, gen_phi, gen_pi_att = parrot.sample_model(
        labels_tr, labels_mask_tr, features_mask_tr,
        speaker_tr, args.num_samples, args.num_steps)

print "Successfully sampled the parrot."

gen_x = gen_x.swapaxes(0, 1)
gen_phi = gen_phi.swapaxes(0, 1)

features_lengths = []
labels_lengths = []
for idx in range(args.num_samples):
    # Heuristic for deciding when to end the sampling.
    this_phi = gen_phi[idx]
    this_labels_length = int(labels_mask_tr[idx].sum())

    try:
        this_features_length = numpy.where((
            this_phi[:, this_labels_length, numpy.newaxis] >
            this_phi[:, :this_labels_length-1]).all(axis=1))[0][0]
        this_features_length = numpy.minimum(
            args.num_steps, this_features_length + 40)  # Small extra time.
    except:
        print "Its better to increase the number of samples."
        this_features_length = args.num_steps

    features_lengths.append(this_features_length)
    labels_lengths.append(this_labels_length)

if saved_args.raw_output:
    print "Sampling and saving raw audio..."

    to_save_path = os.path.join(args.save_dir, 'samples', 'new_raw')
    if not os.path.exists(to_save_path):
        os.makedirs(to_save_path)

    parrot.sampleRnn.sample_raw(gen_x.swapaxes(0, 1).copy(), features_lengths, args.samples_name, to_save_path)
    print "Successfully sampled raw audio..."


norm_info_file = os.path.join(
    data_dir, args.dataset,
    'norm_info_mgc_lf0_vuv_bap_63_MVN.dat')

for idx, this_sample in enumerate(gen_x):
    this_sample = this_sample[:features_lengths[idx]]
    generate_wav(
        this_sample,
        os.path.join(args.save_dir, 'samples'),
        args.samples_name + '_' + str(idx),
        sptk_dir=args.sptk_dir,
        world_dir=args.world_dir,
        norm_info_file=norm_info_file,
        do_post_filtering=args.do_post_filtering)

if args.plot_raw:
    from scipy.io import wavfile
    raw_audio = data_tr['raw_audio'].swapaxes(0, 1)

    for idx in range(args.num_samples):
        this_raw = numpy.concatenate(
            raw_audio[idx])[:80*int(features_mask_tr.sum(axis=0)[idx])]
        wavfile.write(
            os.path.join(
                args.save_dir, 'samples',
                'raw_' + args.samples_name + '_' + str(idx) + '.wav'),
            16000, this_raw)

if args.process_originals:
    assert not args.new_sentences
    for i, this_sample in enumerate(features_tr.swapaxes(0, 1)):
        this_sample = this_sample[:int(features_mask_tr.sum(axis=0)[i])]
        generate_wav(
            this_sample,
            os.path.join(args.save_dir, 'samples'),
            'original_' + args.samples_name + '_' + str(i),
            sptk_dir=args.sptk_dir,
            world_dir=args.world_dir,
            norm_info_file=norm_info_file,
            do_post_filtering=args.do_post_filtering)

all_text = []
for idx in range(args.num_samples):
    this_labels = labels_tr[idx, :labels_lengths[idx]]
    this_phi = gen_phi[idx][:features_lengths[idx], :labels_lengths[idx]]

    this_text = attention_plot(
        this_phi,
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + '_' + str(idx)),
        this_labels,
        args.dataset,
        saved_args.labels_type)
    this_text = ''.join(this_text)
    all_text.append(this_text + '\n')

with open(
    os.path.join(
        args.save_dir, 'samples', args.samples_name + '.txt'),
        'w') as text_file:
    text_file.write(''.join(all_text).encode('utf8'))

if args.debug_plot:
    from utils import full_plot
    gen_k = gen_k.swapaxes(0, 1)
    gen_w = gen_w.swapaxes(0, 1)
    gen_pi = gen_pi.swapaxes(0, 1)
    gen_phi = gen_phi.swapaxes(0, 1)
    gen_pi_att = gen_pi_att.swapaxes(0, 1)

    for i in range(args.num_samples):
        this_num_steps = int(features_mask_tr.sum(axis=0)[i])
        this_labels_length = int(labels_mask_tr.sum(axis=1)[i])
        this_x = gen_x[i][:this_num_steps]
        this_k = gen_k[i][:this_num_steps]
        this_w = gen_w[i][:this_num_steps]
        this_pi = gen_pi[i][:this_num_steps]
        this_phi = gen_phi[i][:this_num_steps, :this_labels_length]
        this_pi_att = gen_pi_att[i][:this_num_steps]

        full_plot(
            [this_x, this_pi_att, this_k, this_w, this_phi],
            os.path.join(
                args.save_dir, 'samples',
                args.samples_name + '_' + str(i) + '.png'))

if args.animation:
    for i in range(args.num_samples):
        this_num_steps = int(features_mask_tr.sum(axis=0)[i])
        this_labels_length = int(labels_mask_tr.sum(axis=1)[i])
        this_x = gen_x[i][:this_num_steps]
        this_k = gen_k[i][:this_num_steps]
        this_w = gen_w[i][:this_num_steps]
        this_pi = gen_pi[i][:this_num_steps]
        this_phi = gen_phi[i][:this_num_steps, :this_labels_length]
        this_pi_att = gen_pi_att[i][:this_num_steps]
        create_animation(
            [this_x, this_pi_att, this_k, this_w, this_phi],
            args.samples_name + '_' + str(i) + '.wav',
            args.samples_name + '_' + str(i),
            os.path.join(args.save_dir, 'samples'))

    if args.process_originals:
        for i in range(args.num_samples):
            this_num_steps = int(features_mask_tr.sum(axis=0)[i])
            this_x = features_tr[:, i][:this_num_steps]
            this_phoneme = labels_tr[:, i][:this_num_steps]
            create_animation(
                [this_x, numpy_one_hot(
                    this_phoneme, saved_args.num_characters)],
                'original_' + args.samples_name + '_' + str(i) + '.wav',
                'original_' + args.samples_name + '_' + str(i),
                os.path.join(args.save_dir, 'samples'))
