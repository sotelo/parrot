import os

from fuel import config
from fuel.schemes import (
    ConstantScheme, ShuffledExampleScheme,
    SequentialExampleScheme)
from fuel.transformers import (
    AgnosticSourcewiseTransformer, Batch, Filter, FilterSources,
    Mapping, Padding, Rename, SortMapping, Transformer, Unpack)
from fuel.streams import DataStream

from fuel.datasets import H5PYDataset

from quantize import __batch_quantize

import numpy


def _length(data):
    return len(data[0])


def _transpose(data):
    return data.swapaxes(0, 1)


def _chunk(data, frame_size=80, axis=1):
    return numpy.stack(numpy.split(data, data.shape[axis]/frame_size, axis))


def _check_batch_size(data, batch_size):
    return len(data[0]) == batch_size


def _check_ratio(data, idx1, idx2, min_val, max_val):
    ratio = len(data[idx1]) / float(len(data[idx2]))
    # print (min_val <= ratio and ratio <= max_val)
    return (min_val <= ratio and ratio <= max_val)


class SegmentSequence(Transformer):
    """Segments the sequences in a batch.

    This transformer is useful to do tbptt. All the sequences to segment
    should have the time dimension as their first dimension.
    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    seq_size : int
        maximum size of the resulting sequences.
    which_sources : tuple of str, optional
        sequences to segment
    add_flag : bool, optional
        add a flag indicating the beginning of a new sequence.
    flag_name : str, optional
        name of the source for the flag
    min_size : int, optional
        smallest possible sequence length for the last cut
    return_last : bool, optional
        return the last cut of the sequence, which might be different size
    share_value : int, optional
        size of overlap
    """

    def __init__(self,
                 data_stream,
                 seq_size=100,
                 which_sources=None,
                 add_flag=False,
                 flag_name=None,
                 min_size=10,
                 return_last=True,
                 share_value=0,
                 **kwargs):

        super(SegmentSequence, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)

        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources

        self.seq_size = seq_size
        self.step = 0
        self.data = None
        self.len_data = None
        self.add_flag = add_flag
        self.min_size = min_size
        self.share_value = share_value

        if not return_last:
            self.min_size += self.seq_size

        if flag_name is None:
            flag_name = u"start_flag"

        self.flag_name = flag_name

    @property
    def sources(self):
        return self.data_stream.sources + ((self.flag_name,)
                                           if self.add_flag else ())

    def get_data(self, request=None):
        flag = 0

        if self.data is None:
            self.data = next(self.child_epoch_iterator)
            idx = self.sources.index(self.which_sources[0])
            self.len_data = self.data[idx].shape[0]
            flag = 1  # if flag is here: first part

        segmented_data = list(self.data)

        for source in self.which_sources:
            idx = self.sources.index(source)
            # Segment data:
            segmented_data[idx] = self.data[idx][
                self.step:(self.step + self.seq_size)]

        self.step += self.seq_size

        # Size of overlap:
        self.step -= self.share_value

        if self.step + self.min_size >= self.len_data:
            self.data = None
            self.len_data = None
            self.step = 0
            # flag = 1  # if flag is here: last part

        if self.add_flag:
            segmented_data.append(flag)

        return tuple(segmented_data)


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


class AddConstantSource(Mapping):
    def __init__(self, data_stream, constant, name, **kwargs):
        super(AddConstantSource, self).__init__(
            data_stream, lambda x: (constant,), (name,), **kwargs)


class VoiceData(H5PYDataset):
    def __init__(self, voice, which_sets, filename=None, **kwargs):

        assert voice in [
            'arctic', 'blizzard', 'dimex', 'librispeech', 'pavoque', 'vctk']

        self.voice = voice

        if filename is None:
            filename = voice

        self.filename = filename + '.hdf5'
        super(VoiceData, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], self.voice, self.filename)


def get_raw_transformer(q_type, q_level):
    def transformer(batch):
        # import ipdb; ipdb.set_trace()
        batch_shape = batch.shape
        batch = batch.transpose(1, 0, 2).reshape((batch_shape[1], -1))
        batch = __batch_quantize(batch, q_level, q_type)
        batch = batch.reshape((batch_shape[1], -1, 80))
        batch = batch.transpose(1,0,2)
        return batch
    return transformer


def parrot_stream(
        voice, use_speaker=False, which_sets=('train',), batch_size=32,
        seq_size=50, num_examples=None, sorting_mult=4, noise_level=None,
        labels_type='full_labels', check_ratio=False, raw_data=True, q_type='mu-law', q_level=256):

    assert labels_type in [
        'full_labels', 'phonemes', 'unconditional',
        'unaligned_phonemes', 'text']

    dataset = VoiceData(voice=voice, which_sets=which_sets)

    sorting_size = batch_size * sorting_mult

    if not num_examples:
        num_examples = dataset.num_examples

    if 'train' in which_sets:
        scheme = ShuffledExampleScheme(num_examples)
    else:
        scheme = SequentialExampleScheme(num_examples)

    data_stream = DataStream.default_stream(dataset, iteration_scheme=scheme)

    if check_ratio and labels_type in ['unaligned_phonemes', 'text']:
        idx = data_stream.sources.index(labels_type)
        min_val = 8 if labels_type == 'text' else 12.
        max_val = 16 if labels_type == 'text' else 25.
        data_stream = Filter(
            data_stream, lambda x: _check_ratio(x, 0, idx, min_val, max_val))

    segment_sources = ('features', 'features_mask')
    all_sources = segment_sources

    if raw_data:
        raw_sources = ('raw_audio', )
        all_sources += raw_sources
    else:
        raw_sources = ()

    if labels_type != 'unconditional':
        all_sources += ('labels', )
        data_stream = Rename(data_stream, {labels_type: 'labels'})

    if labels_type in ['full_labels', 'phonemes']:
        segment_sources += ('labels',)

    elif labels_type in ['unaligned_phonemes', 'text']:
        all_sources += ('labels_mask', )

    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(sorting_size))
    data_stream = Mapping(data_stream, SortMapping(_length))
    data_stream = Unpack(data_stream)
    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(batch_size))

    data_stream = Filter(
        data_stream, lambda x: _check_batch_size(x, batch_size))

    data_stream = Padding(data_stream)

    if use_speaker:
        data_stream = FilterSources(
            data_stream, all_sources + ('speaker_index',))
    else:
        data_stream = FilterSources(
            data_stream, all_sources)

    data_stream = SourceMapping(
        data_stream, _transpose, which_sources=segment_sources)

    # The conditional is not necessary, but I'm still adding it for clarity.
    if raw_data:
        data_stream = SourceMapping(
            data_stream, _chunk, which_sources=raw_sources)

        raw_transformer = get_raw_transformer(q_type, q_level)
        data_stream = SourceMapping(
            data_stream, raw_transformer, which_sources=raw_sources)

    data_stream = SegmentSequence(
        data_stream,
        seq_size=seq_size + 1,
        share_value=1,
        return_last=False,
        add_flag=True,
        which_sources=segment_sources + raw_sources)

    if noise_level is not None:
        data_stream = AddConstantSource(
            data_stream, noise_level, 'feedback_noise_level')

    return data_stream

if __name__ == "__main__":
    data_stream = parrot_stream(
        'dimex', labels_type='text', seq_size=10,
        batch_size=10, sorting_mult=1, check_ratio=False, raw_data=True)
    print data_stream.sources
    data_tr = next(data_stream.get_epoch_iterator())

    for idx, source in enumerate(data_stream.sources):
        if source not in ['start_flag', 'feedback_noise_level']:
            print source, "shape: ", data_tr[idx].shape, \
                source, "dtype: ", data_tr[idx].dtype
        else:
            print source, ": ", data_tr[idx]


    # print next(data_stream.get_epoch_iterator())[-1]
    # import ipdb; ipdb.set_trace()
    # # For Arctic, the ratio is 18 steps of features per letter.
    # data_tr = next(data_stream.get_epoch_iterator())
    # ratios = (data_tr[1].sum(0) / data_tr[3].sum(1))
    # print numpy.percentile(ratios, [0, 10, 25, 50, 75, 90, 99, 100])

    # Arctic
    # phonemes: array([ 12.84, 14.75, 15.56, 16.82, 18.16, 19.89, 48.8])
    # text:     array([  8.2, 9.89, 10.39, 11.07, 11.91, 12.81, 24.4])

    # Blizzard
    # phonemes: array([ 6.26, 14.07, 15.11, 16.26, 17.60, 19.23, 103.33])
    # text:     array([4.37, 9.8, 10.64, 11.62, 12.59, 13.76, 46. ])

    # VCTK
    # phonemes: array([  3., 12.39, 13.52, 15.03, 16.8, 18.96, 40.5])
    # text:     array([  2.04, 8.43, 9.23, 10.28, 11.56, 13.03, 23.15])
