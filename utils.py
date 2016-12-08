"""Utils for training and sampling.

This file contains the arguments parser.
"""
import argparse
import os
import matplotlib
import numpy

from subprocess import call
matplotlib.use('Agg')
from matplotlib import animation, pyplot, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

SAMPLING_RATE = 16000
REFRESH_RATE = 100  # images per second

SPTK_DIR = '/Tmp/sotelo/code/merlin/tools/bin/SPTK-3.9/'
WORLD_DIR = '/Tmp/sotelo/code/merlin/tools/bin/WORLD/'


def numpy_one_hot(data, n_class=None):
    if n_class is None:
        n_class = data.max() + 1
    return numpy.eye(n_class)[data]


def plot_matrix(data, ax):
    im = ax.imshow(
        data.T, aspect='auto', origin='lower', interpolation='nearest')
    cax = make_axes_locatable(ax).append_axes("right", size="1%", pad=0.05)
    cb = pyplot.colorbar(im, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(ticker.AutoLocator())
    cb.update_ticks()


def anim_to_mp4(anim, audio_file, filename, save_dir):
    writer = animation.writers['ffmpeg']
    writer = writer(fps=REFRESH_RATE, bitrate=SAMPLING_RATE)

    filename = os.path.join(save_dir, filename)

    if not hasattr(anim, '_encoded_video'):
        anim.save(filename + '_temp.mp4', writer=writer)
        call(['rm', filename + '.mp4'])
        call([
            'ffmpeg', '-i', filename + '_temp.mp4', '-i',
            os.path.join(save_dir, audio_file), filename + '.mp4'])
        call(['rm', filename + '_temp.mp4'])


def full_plot(matrices, save_name):
    n_plots = len(matrices)

    f, axarr = pyplot.subplots(n_plots, 1)

    for idx, mat in enumerate(matrices):
        plot_matrix(mat, axarr[idx])

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print "Error building image!: " + save_name

    pyplot.close()


def create_animation(matrices, audio_file, filename, save_dir):
    step = SAMPLING_RATE / REFRESH_RATE
    frames = 80 * len(matrices[0]) / step
    interval = 1000 / REFRESH_RATE

    n_plots = len(matrices)

    fig, axarr = pyplot.subplots(n_plots, 1)

    for arr in axarr:
        arr.set_xlim([0, len(matrices[0])])

    lines = []
    for arr in axarr:
        vline = arr.axvline(x=0., color='k')
        lines.append(vline)

    def init():
        for idx, mat in enumerate(matrices):
            plot_matrix(mat, axarr[idx])

    def animate(n_step):
        for idx in range(n_plots):
            lines[idx].set_xdata(step * n_step / 80)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=interval, blit=True)

    anim_to_mp4(anim, audio_file, filename, save_dir)


def train_parse():
    """Parser for training arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--input_dim', type=int, default=420,
                        help='dimension of labels')
    parser.add_argument('--output_dim', type=int, default=63,
                        help='dimension of output')
    parser.add_argument('--rnn_h_dim', type=int, default=1024,
                        help='size of time wise RNN hidden state')
    parser.add_argument('--readouts_dim', type=int, default=1024,
                        help='size of readouts')
    parser.add_argument('--weak_feedback', type=bool, default=False,
                        help='feedback to top layer')
    parser.add_argument('--full_feedback', type=bool, default=False,
                        help='feedback to all layers')
    parser.add_argument('--feedback_noise_level', type=float, default=None,
                        help='how much noise in the feedback from audio')
    parser.add_argument('--layer_norm', type=bool, default=False,
                        help='use simple layer normalization')
    parser.add_argument('--labels_type', type=str, default='full_labels',
                        help='which kind of labels to use: full or phoneme')
    parser.add_argument('--which_cost', type=str, default='MSE',
                        help='which cost to use MSE or GMM')
    parser.add_argument('--attention_type', type=str, default='graves',
                        help='which attention to use')
    parser.add_argument('--attention_alignment', type=float, default=1.,
                        help='bias the alignment')
    parser.add_argument('--num_characters', type=int, default=43,
                        help='how many characters in the labels dict')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
    parser.add_argument('--seq_size', type=int, default=50,
                        help='length of the training sequences')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
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
    parser.add_argument('--adaptive_noise', type=bool,
                        default=False,
                        help='whether to use adaptive noise')
    parser.add_argument('--load_experiment', type=str,
                        default=None,
                        help='name of the experiment that will be loaded')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='time in hours that the model will run')
    parser.add_argument('--use_speaker', type=bool,
                        default=False,
                        help='use speaker conditioning information')
    parser.add_argument('--num_speakers', type=int,
                        default=22,
                        help='adam or adasecant')
    parser.add_argument('--speaker_dim', type=int,
                        default=128,
                        help='adam or adasecant')
    parser.add_argument('--dataset', type=str,
                        default='vctk',
                        help='which dataset to use')
    parser.add_argument('--save_dir', type=str,
                        default=os.environ['RESULTS_DIR'],
                        help='save dir directory')

    args = parser.parse_args()
    if args.dataset not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.dataset)

    if args.adaptive_noise:
        args.batch_size = 1

    return args


def sample_parse():
    """Parser for sampling arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--sampling_bias', type=float, default=1.,
                        help='the higher the bias the smoother the samples')
    parser.add_argument('--timing_coeff', type=float, default=1.,
                        help='make attention go faster or slower')
    parser.add_argument('--sharpening_coeff', type=float, default=1.,
                        help='reduce variance of attention gaussians')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples')
    parser.add_argument('--num_steps', type=int, default=2048,
                        help='maximum size of each sample')
    parser.add_argument('--samples_name', type=str, default='sample',
                        help='name to save the samples.')
    parser.add_argument('--speaker_id', type=int, default=None,
                        help='which speaker voice')
    parser.add_argument('--mix', type=float, default=None,
                        help='mix between two voices')
    parser.add_argument('--dataset', type=str,
                        default='vctk',
                        help='which dataset to use')
    parser.add_argument('--new_sentences', type=bool,
                        default=False,
                        help='Generate new sentences or sentences from valid')
    parser.add_argument('--save_dir', type=str,
                        default=os.environ['RESULTS_DIR'],
                        help='save dir directory')
    parser.add_argument('--sptk_dir', type=str,
                        default=SPTK_DIR,
                        help='save dir directory')
    parser.add_argument('--world_dir', type=str,
                        default=WORLD_DIR,
                        help='save dir directory')
    parser.add_argument('--process_originals', type=bool,
                        default=False,
                        help='Process examples from the dataset or not')
    parser.add_argument('--do_post_filtering', type=bool,
                        default=False,
                        help='do post filtering process')
    parser.add_argument('--animation', type=bool,
                        default=False,
                        help='wether to do animation or no')
    parser.add_argument('--sample_one_step', type=bool,
                        default=False,
                        help='wether to only sample one step or all')
    parser.add_argument('--use_last', type=bool,
                        default=False,
                        help='wether to use the best parameters or last')
    parser.add_argument('--phrase', type=str,
                        default=None,
                        help='which phrase to generate')
    parser.add_argument('--random_speaker', type=bool,
                        default=False,
                        help='generate with random speaker')

    args = parser.parse_args()
    if args.dataset not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.dataset)

    return args
