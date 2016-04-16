import argparse
import numpy
import os

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


def train_parse():
    """Parser for training arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--num_freq', type=int, default=257,
                        help='number of frequencies in the spectrum')
    parser.add_argument('--rnn1_size', type=int, default=400,
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
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='maximum size of each sample')
    parser.add_argument('--model', type=str, default='full',
                        help='type of model')
    # args = parser.parse_args()
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
    parser.add_argument('--save_dir', type=str,
                        #default='./trained/',
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument(
        '--phrase', type=str,
        default='WHAT SHOULD I SAY TO PROVE THAT I AM CAPABLE OF SPEAKING \
        HOW CAN I PROVE THAT I CAN BE TRUSTED TO SPEAK FOR YOU BY MYSELF',
        help='phrase to write')
    parser.add_argument('--one_step_sampling', type=bool, default=False,
                        help='sample only one step')
    return parser
