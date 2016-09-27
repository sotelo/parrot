import argparse
import numpy
import pysptk
import subprocess
import os
import tempfile

from scipy.io import wavfile
from scipy.signal import lfilter

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

h_filters = h_filters = numpy.load('h_filters.npy')

save_dir = os.environ['RESULTS_DIR']
if 'blizzard' not in save_dir:
    save_dir = os.path.join(save_dir, 'blizzard/')

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'full_standardize.npz')
data_stats = numpy.load(data_dir)

mean_f0 = data_stats['mean_f0']
mean_mgc = data_stats['mean_mgc']
mean_spectrum = data_stats['mean_spectrum']
mean_voicing_str = data_stats['mean_voicing_str']

std_f0 = data_stats['std_f0']
std_mgc = data_stats['std_mgc']
std_spectrum = data_stats['std_spectrum']
std_voicing_str = data_stats['std_voicing_str']

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'blizzard_limits.npz')
data_stats = numpy.load(data_dir)

audio_len_lower_limit = data_stats['audio_len_lower_limit']
audio_len_upper_limit = data_stats['audio_len_upper_limit']
transcripts_len_lower_limit = data_stats['transcripts_len_lower_limit']
transcripts_len_upper_limit = data_stats['transcripts_len_upper_limit']
attention_proportion_lower_limit = data_stats['attention_proportion_lower_limit']
attention_proportion_upper_limit = data_stats['attention_proportion_upper_limit']
mgc_lower_limit = data_stats['mgc_lower_limit']
mgc_upper_limit = data_stats['mgc_upper_limit']
spectrum_lower_limit = data_stats['spectrum_lower_limit']
spectrum_upper_limit = data_stats['spectrum_upper_limit']
voiced_proportion_lower_limit = data_stats['voiced_proportion_lower_limit']
voiced_proportion_upper_limit = data_stats['voiced_proportion_upper_limit']
min_voiced_lower_limit = data_stats['min_voiced_lower_limit']
min_voiced_upper_limit = data_stats['min_voiced_upper_limit']
max_voiced_lower_limit = data_stats['max_voiced_lower_limit']

all_chars = ([chr(ord('A') + i) for i in range(26)] + [' ', '<UNK>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'


def mgcf02wav(
        mgc,
        f0,
        order=34,
        shift_window=64,
        pass_const=0.4,
        mgcep_gamma=2,
        gaussian=False):
    mgc = mgc.astype('float32')
    f0 = f0.astype('float32')

    with tempfile.NamedTemporaryFile() as f:
        mgc_fix_cmd = (
            'mgc2mgclsp -m {} -a {} -c {} -s 16000 | '
            'lspcheck -m {} -c -r 0.01 | '
            'mgclsp2mgc -m {} -a {} -c {}').format(
            order, pass_const, mgcep_gamma, order, order,
            pass_const, mgcep_gamma)

        p = subprocess.Popen(
            mgc_fix_cmd, stdout=f, stdin=subprocess.PIPE, shell=True)
        mgc_fix, stderr = p.communicate(mgc.ravel().tobytes())
        f.file.flush()
        f.file.close()

        excitation_cmd = 'excite -p {}'.format(shift_window)
        p = subprocess.Popen(
            excitation_cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True)
        excitation, stderr = p.communicate(f0.tobytes())

        synthesis_cmd = 'mglsadf -m {} -a {} -c {} -p {} {}'.format(
            order, pass_const, mgcep_gamma, shift_window, f.name)
        p = subprocess.Popen(
            synthesis_cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True)
        stdout, stderr = p.communicate(excitation)
        y = numpy.fromstring(stdout, dtype='float32')

    return y


def mgcf0sp2wav(
        mgc,
        f0,
        voicing_str,
        order=34,
        shift_window=64,
        pass_const=0.4,
        mgcep_gamma=2,
        gaussian=False):
    mgc = mgc.astype('float32')
    f0 = f0.astype('float32')

    with tempfile.NamedTemporaryFile() as f:
        mgc_fix_cmd = (
            'mgc2mgclsp -m {} -a {} -c {} -s 16000 | '
            'lspcheck -m {} -c -r 0.01 | '
            'mgclsp2mgc -m {} -a {} -c {}').format(
            order, pass_const, mgcep_gamma, order, order,
            pass_const, mgcep_gamma)

        p = subprocess.Popen(
            mgc_fix_cmd, stdout=f, stdin=subprocess.PIPE, shell=True)
        mgc_fix, stderr = p.communicate(mgc.ravel().tobytes())
        f.file.flush()
        f.file.close()

        excitation = mixed_excitation(
            f0, voicing_str, shift_window).astype('float32')

        synthesis_cmd = 'mglsadf -m {} -a {} -c {} -p {} {}'.format(
            order, pass_const, mgcep_gamma, shift_window, f.name)
        p = subprocess.Popen(
            synthesis_cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True)
        stdout, stderr = p.communicate(excitation.tobytes())
        y = numpy.fromstring(stdout, dtype='float32')

    return y


def mixed_excitation(f0, voicing_str, hopsize):
    exc_voiced = pysptk.excite(
        f0.astype(numpy.float64), hopsize=hopsize, noise=False)
    exc_unvoiced = 2 * numpy.random.rand(len(exc_voiced)) - 1

    exc = numpy.zeros(len(exc_voiced))

    for i in range(5):
        h = h_filters[i]
        x_v = lfilter(h, 1, exc_voiced)
        x_uv = lfilter(h, 1, exc_unvoiced)

        gain_v = numpy.zeros(len(exc_voiced))
        gain_uv = numpy.zeros(len(exc_voiced))

        str_v = voicing_str[:, i]
        for k in range(len(str_v)):
            if f0[k] > 0:
                gain_v[k * hopsize:(k + 1) * hopsize] = str_v[k]
                gain_uv[k * hopsize:(k + 1) * hopsize] = 1.0 - str_v[k]
            else:
                gain_v[k * hopsize:(k + 1) * hopsize] = 0.0
                gain_uv[k * hopsize:(k + 1) * hopsize] = 1.0

        exc += (gain_v * x_v + gain_uv * x_uv)

    return exc


def plot_sample(spectrum, f0, pi, phi, pi_att, fig_name):
    f, axarr = pyplot.subplots(5, sharex=True)
    f.set_size_inches(10, 8.5)
    im0 = axarr[0].imshow(
        spectrum.T, origin='lower',
        aspect='auto', interpolation='nearest')
    axarr[1].plot(f0, linewidth=1)

    im2 = axarr[2].imshow(
        pi.T, origin='lower',
        aspect='auto', interpolation='nearest')
    im3 = axarr[3].imshow(
        phi.T, origin='lower',
        aspect='auto', interpolation='nearest')
    im4 = axarr[4].imshow(
        pi_att.T, origin='lower',
        aspect='auto', interpolation='nearest')

    cax0 = make_axes_locatable(axarr[0]).append_axes(
        "right", size="1%", pad=0.05)
    cax2 = make_axes_locatable(axarr[2]).append_axes(
        "right", size="1%", pad=0.05)
    cax3 = make_axes_locatable(axarr[3]).append_axes(
        "right", size="1%", pad=0.05)
    cax4 = make_axes_locatable(axarr[4]).append_axes(
        "right", size="1%", pad=0.05)

    pyplot.colorbar(im0, cax=cax0)
    pyplot.colorbar(im2, cax=cax2)
    pyplot.colorbar(im3, cax=cax3)
    pyplot.colorbar(im4, cax=cax4)

    pyplot.savefig(fig_name)
    pyplot.close()


def spectrum_to_audio(spectrum, f0, use_spectrum, file_name):
    if use_spectrum:
        spectrum = numpy.hstack([spectrum, spectrum[:, ::-1][:, 1:-1]])
        spectrum = spectrum.astype('float64').copy(order='C')
        mgc = numpy.apply_along_axis(
            pysptk.mgcep, 1, spectrum, order, alpha, gamma,
            eps=0.0012, etype=1, itype=2)
    else:
        mgc = spectrum
    x_synth = mgcf02wav(mgc, f0)
    x_synth = .95 * x_synth / max(abs(x_synth)) * 2**15
    wavfile.write(file_name, 16000, x_synth.astype('int16'))


def train_parse():
    """Parser for training arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--num_freq', type=int, default=257,
                        help='number of frequencies in the spectrum')
    parser.add_argument('--rnn1_size', type=int, default=1000,
                        help='size of time wise RNN hidden state')
    parser.add_argument('--rnn2_size', type=int, default=100,
                        help='size of frequency wise RNN hidden state')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_mixture', type=int, default=20,
                        help='number of gaussian mixtures for the spectrum')
    parser.add_argument('--k_f0', type=int, default=5,
                        help='number of gaussian mixtures for f0')
    parser.add_argument('--save_dir', type=str,
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument('--size_attention', type=int, default=10,
                        help='number of normal components for attention')
    parser.add_argument('--num_letters', type=int, default=28,
                        help='size of dictionary')
    parser.add_argument('--model', type=str, default='simple',
                        help='type of model')
    parser.add_argument('--platoon_port', type=int,
                        default=None,
                        help='port where platoon server is running')
    parser.add_argument('--algorithm', type=str,
                        default='adam',
                        help='adam or adasecant')
    parser.add_argument('--grad_clip', type=float,
                        default=0.9,
                        help='how much to clip the gradients. for adam is 10x')
    parser.add_argument('--lr_schedule', type=bool,
                        default=False,
                        help='whether to use the learning rate schedule')
    parser.add_argument('--load_experiment', type=str,
                        default=None,
                        help='name of the experiment that will be loaded')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='time in hours that the model will run')
    parser.add_argument('--attention_type', type=str,
                        default='softmax',
                        help='graves or softmax')
    parser.add_argument('--attention_alignment', type=float,
                        default=0.05,
                        help='initial lengths of each attention step')
    parser.add_argument('--use_spectrum', type=bool,
                        default=False,
                        help='use spectrum or mgc')
    return parser


def train_raw_parse():
    """Parser for training arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--frame_size', type=int, default=4)
    parser.add_argument('--rnn1_size', type=int, default=1000,
                        help='size of time wise RNN hidden state')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=64 * 4,
                        help='RNN sequence length')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--save_dir', type=str,
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument('--num_levels', type=int, default=256,
                        help='size of quantization')
    parser.add_argument('--platoon_port', type=int,
                        default=None,
                        help='port where platoon server is running')
    parser.add_argument('--algorithm', type=str,
                        default='adam',
                        help='adam or adasecant')
    parser.add_argument('--grad_clip', type=float,
                        default=0.9,
                        help='how much to clip the gradients. for adam is 10x')
    parser.add_argument('--lr_schedule', type=bool,
                        default=False,
                        help='whether to use the learning rate schedule')
    parser.add_argument('--load_experiment', type=str,
                        default=None,
                        help='name of the experiment that will be loaded')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='time in hours that the model will run')
    return parser


def sample_parse():
    """Parser for sampling arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--sampling_bias', type=float, default=1.,
                        help='the higher the bias the smoother the samples')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples')
    parser.add_argument('--num_steps', type=int, default=2048,
                        help='maximum size of each sample')
    parser.add_argument('--samples_name', type=str, default='sample',
                        help='name to save the samples.')
    parser.add_argument('--save_dir', type=str,
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument(
        '--phrase', type=str,
        default='WHAT SHOULD I SAY TO PROVE THAT I AM CAPABLE OF SPEAKING \
        HOW CAN I PROVE THAT I CAN BE TRUSTED TO SPEAK FOR YOU BY MYSELF',
        help='phrase to write')
    return parser
