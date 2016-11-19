"""Utils for training and sampling.

This file contains the arguments parser.
"""
import argparse
import os


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
    parser.add_argument('--layer_normalization', type=bool, default=False,
                        help='use simple layer normalization')
    parser.add_argument('--which_cost', type=str, default='MSE',
                        help='which cost to use MSE or GMM')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
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
    parser.add_argument('--load_experiment', type=str,
                        default=None,
                        help='name of the experiment that will be loaded')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='time in hours that the model will run')
    parser.add_argument('--use_speaker', type=bool,
                        default=True,
                        help='use speaker conditioning information')
    parser.add_argument('--num_speakers', type=int,
                        default=21,
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

    args = parser.parse_args()
    if args.dataset not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.dataset)

    return args
