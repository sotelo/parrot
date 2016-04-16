import os

from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from fuel.transformers import (
    AgnosticSourcewiseTransformer, Batch, Filter, FilterSources, ForceFloatX,
    Mapping, Padding, ScaleAndShift, SortMapping, Unpack)
from fuel.streams import DataStream

from play.toy.segment_transformer import SegmentSequence
from play.datasets.blizzard import Blizzard

import numpy

from utils import (
    mean_f0, mean_mgc, mean_spectrum, std_f0, std_mgc, std_spectrum,
    audio_len_lower_limit, audio_len_upper_limit, transcripts_len_lower_limit,
    transcripts_len_upper_limit, attention_proportion_lower_limit,
    attention_proportion_upper_limit, mgc_lower_limit, mgc_upper_limit,
    spectrum_lower_limit, spectrum_upper_limit, voiced_proportion_lower_limit,
    voiced_proportion_upper_limit, min_voiced_lower_limit,
    min_voiced_upper_limit, max_voiced_lower_limit)


def _length(data):
    return len(data[0])


def _remove_nans(data):
    if numpy.any(numpy.isnan(data[-1])):
        f0, mgc, spectrum, transcripts, voicing_str = data
        idx = numpy.where(numpy.any(numpy.isnan(voicing_str), axis=1))[0][0]
        data = tuple([f0[:idx], mgc[:idx], spectrum[:idx],
                      transcripts, voicing_str[:idx]])
    return data


def _equalize_length(data):
    f0, f0_mask, mgc, spectrum, \
        transcripts, transcripts_mask, voicing_str = data
    idx = int(f0_mask[0].sum())
    return tuple([f0[:, :idx], f0_mask[:, :idx], mgc[:, :idx],
                  spectrum[:, :idx], transcripts, transcripts_mask,
                  voicing_str[:, :idx]])


def _transpose(data):
    return data.swapaxes(0, 1)


def _is_nonzero(data):
    return tuple([1. * (data[0] > 0)])


def _zero_for_unvoiced(data):
    # Multiply by 0 the unvoiced components. HARDCODED
    return tuple((data[0] * data[-1],) + data[1:])


def _clip_f0(data, ceil=300.):
    temp_var = data[0]
    temp_var[temp_var > ceil] = ceil
    return tuple((temp_var,) + data[1:])


def _filter_blizzard(data):
    f0, mgc, spectrum, transcripts, voicing_str = data

    len_f0 = len(f0)
    len_transcripts = len(transcripts)
    attention_proportion = len_f0 / float(len_transcripts)
    min_mgc = mgc.min(axis=0)
    max_mgc = mgc.max(axis=0)
    min_spectrum = spectrum.min(axis=0)
    max_spectrum = spectrum.max(axis=0)
    voiced_proportion = (f0 > 0).mean()
    min_voiced = f0[f0 > 0].min()
    max_voiced = f0[f0 > 0].max()

    return len_f0 >= audio_len_lower_limit and \
        len_f0 <= audio_len_upper_limit and \
        len_transcripts >= transcripts_len_lower_limit and \
        len_transcripts <= transcripts_len_upper_limit and \
        attention_proportion >= attention_proportion_lower_limit and \
        attention_proportion <= attention_proportion_upper_limit and \
        (min_mgc >= mgc_lower_limit).all() and \
        (max_mgc <= mgc_upper_limit).all() and \
        (min_spectrum >= spectrum_lower_limit).all() and \
        (max_spectrum <= spectrum_upper_limit).all() and \
        voiced_proportion >= voiced_proportion_lower_limit and \
        voiced_proportion <= voiced_proportion_upper_limit and \
        min_voiced >= min_voiced_lower_limit and \
        min_voiced <= min_voiced_upper_limit and \
        max_voiced >= max_voiced_lower_limit and \
        not numpy.isnan(voicing_str).any()


def _chech_batch_size(data, batch_size):
    return len(data[0]) == batch_size


class SourceMapping(AgnosticSourcewiseTransformer):
    """Apply a function to a subset of sources.
    Similar to the Mapping transformer but for a subset of sources.
    It will apply the same function to each source.
    Parameters
    ----------
    mapping : callable

    """

    def __init__(self, data_stream, mapping, **kwargs):
        """Initialization.
        Parameters:
            data_stream: DataStream
            mapping: callable object
        """
        self.mapping = mapping
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(SourceMapping, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(self.mapping(source_data))


def blizzard_stream(which_sets=('train',), batch_size=64,
                    seq_length=100, num_examples=None, sorting_mult=20,
                    which_sources=None):

    all_sources = ('f0', 'f0_mask', 'mgc', 'spectrum',
                   'transcripts', 'transcripts_mask', 'voicing_str',)

    if not which_sources:
        # which_sources = ('f0', 'spectrum', 'start_flag', 'voiced')
        which_sources = ('f0', 'f0_mask', 'spectrum', 'start_flag',
                         'voiced', 'transcripts', 'transcripts_mask')

    dataset = Blizzard(
        which_sets=which_sets, filename="mgc_blizzard_sentence.hdf5")
    sorting_size = batch_size * sorting_mult

    if not num_examples:
        # num_examples = sorting_size * (dataset.num_examples / sorting_size)
        num_examples = dataset.num_examples

    data_stream = DataStream(
        dataset, iteration_scheme=ShuffledExampleScheme(num_examples))

    data_stream = Filter(data_stream, _filter_blizzard)

    data_stream = Mapping(data_stream, _clip_f0)

    # epoch_iterator = data_stream.get_epoch_iterator()

    # means = []
    # stds = []
    # for i in range(num_examples):
    #     print i
    #     f0, mgc, spectrum, transcripts, voicing_str = next(epoch_iterator)

    #     means.append([
    #         f0[f0>0].mean(),
    #         mgc.mean(axis=0),
    #         spectrum.mean(axis=0),
    #         voicing_str.mean(axis=0)
    #         ])

    #     stds.append([
    #         f0[f0>0].std(),
    #         mgc.std(axis=0),
    #         spectrum.std(axis=0),
    #         voicing_str.std(axis=0)
    #         ])

    # means = [numpy.array(mean_).mean(axis = 0) for mean_ in zip(*means)]
    # stds = [numpy.array(std_).mean(axis = 0) for std_ in zip(*stds)]

    # mean_f0, mean_mgc, mean_spectrum, mean_voicing_str = means
    # std_f0, std_mgc, std_spectrum, std_voicing_str = stds

    # data_dir = os.environ['FUEL_DATA_PATH']
    # data_dir = os.path.join(data_dir, 'blizzard/', 'full_standardize.npz')

    # numpy.savez(data_dir,
    #     mean_f0 = mean_f0,
    #     mean_mgc = mean_mgc,
    #     mean_spectrum = mean_spectrum,
    #     mean_voicing_str = mean_voicing_str,
    #     std_f0 = std_f0,
    #     std_mgc = std_mgc,
    #     std_spectrum = std_spectrum,
    #     std_voicing_str = std_voicing_str)

    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(sorting_size))
    data_stream = Mapping(data_stream, SortMapping(_length))
    data_stream = Unpack(data_stream)
    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(batch_size))

    # Cut all sequences to the shape of the smallest one in the batch.
    # So we will not need masks.

    data_stream = Filter(
        data_stream, lambda x: _chech_batch_size(x, batch_size))

    data_stream = Padding(data_stream)
    data_stream = FilterSources(data_stream, all_sources)
    data_stream = Mapping(data_stream, _equalize_length)
    data_stream = FilterSources(
        data_stream,
        ('f0', 'f0_mask', 'mgc', 'spectrum', 'transcripts',
         'transcripts_mask', 'voicing_str'))
    data_stream = SourceMapping(
        data_stream, _transpose,
        which_sources=('f0', 'f0_mask', 'mgc', 'spectrum', 'voicing_str'))
    data_stream = SegmentSequence(
        data_stream,
        seq_length + 1,
        return_last=False,
        which_sources=('f0', 'f0_mask', 'mgc', 'spectrum', 'voicing_str'),
        add_flag=True,
        share_value=True)

    data_stream = Mapping(
        data_stream, _is_nonzero, add_sources=('voiced',))

    data_stream = ScaleAndShift(
        data_stream,
        scale=1 / std_spectrum,
        shift=-mean_spectrum / std_spectrum,
        which_sources=('spectrum',))

    data_stream = ScaleAndShift(
        data_stream,
        scale=1 / std_f0,
        shift=-mean_f0 / std_f0,
        which_sources=('f0',))

    data_stream = ScaleAndShift(
        data_stream,
        scale=1 / std_mgc,
        shift=-mean_mgc / std_mgc,
        which_sources=('mgc',))

    data_stream = Mapping(data_stream, _zero_for_unvoiced)
    data_stream = ForceFloatX(
        data_stream,
        which_sources=('f0', 'f0_mask', 'spectrum', 'start_flag',
                       'voiced', 'transcripts_mask'))

    data_stream = FilterSources(data_stream, which_sources)
    return data_stream

if __name__ == "__main__":
    train_stream = blizzard_stream(batch_size=64, sorting_mult=20)
    print next(train_stream.get_epoch_iterator())
    #import ipdb; ipdb.set_trace()