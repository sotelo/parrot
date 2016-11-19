import matplotlib
import numpy
from time import gmtime, strftime, time

from blocks.extensions import SimpleExtension
from blocks.extensions.saveload import LOADED_FROM
from blocks.filter import VariableFilter
from blocks.roles import ALGORITHM_BUFFER
from blocks.serialization import load_parameters
from pandas import DataFrame

matplotlib.use('Agg')
from matplotlib import pyplot


class Plot(SimpleExtension):
    """Alternative plot extension for blocks.

    Parameters
    ----------
    document : str
        The name of the plot file. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    """

    def __init__(self, document, channels, email=True, **kwargs):
        self.plots = {}
        self.document = document
        self.num_plots = len(channels)
        self.channels = channels
        self.all_channels = list(set([x for small in channels for x in small]))
        self.document = document
        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        df = DataFrame.from_dict(log, orient='index')
        df = df[self.all_channels].astype(float)
        df = df.interpolate('index')

        fig, axarr = pyplot.subplots(self.num_plots, sharex=True)

        if self.num_plots > 1:
            for i, channel in enumerate(self.channels):
                df[channel].plot(ax=axarr[i])
        else:
            df[self.channels[0]].plot()

        pyplot.savefig(self.document)
        pyplot.close()


class TimedFinish(SimpleExtension):
    """Finish training on schedule.

    This extension finishes the training after a certain amount of time. This
    is useful for running in clusters. The time_limit is interpreted as hours.

    """

    def __init__(self, time_limit):
        super(TimedFinish, self).__init__(after_batch=True)
        self.time_limit = time_limit * 60 * 60
        self.start_time = time()

        print "Training started at: ", strftime(
            "%Y-%m-%d %H:%M:%S", gmtime(self.start_time))

        print "Training will be finished at: ", strftime(
            "%Y-%m-%d %H:%M:%S", gmtime(self.start_time + self.time_limit))

    def do(self, which_callback, *args):
        if time() - self.start_time > self.time_limit:
            self.main_loop.log.current_row['training_finish_requested'] = True


class LearningRateSchedule(SimpleExtension):
    """Control learning rate.

    This extensions decreases the learning rate when the validation error
    flattens. It also takes care of early stopping.

    """

    def __init__(
            self,
            lr,
            track_var,
            path,
            patience=5,
            num_cuts=3,
            cut_size=.5, **kwargs):

        self.lr = lr
        self.track_var = track_var
        self.patience = patience
        self.counter = 0
        self.num_cuts = num_cuts
        self.count_cuts = 0
        self.cut_size = cut_size
        self.best_value = numpy.inf
        self.path = path
        self.algorithm_buffers = None

        super(LearningRateSchedule, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        current_value = self.main_loop.log.current_row.get(self.track_var)
        if current_value is None:
            return

        if current_value < self.best_value:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        # If nan, skip all the steps necessary to update the weights.
        if numpy.isnan(current_value):
            self.counter = self.patience + 1

        if self.algorithm_buffers is None:
            self.algorithm_buffers = [
                x for x, y in self.main_loop.algorithm.step_rule_updates]
            self.algorithm_buffers = VariableFilter(
                roles=[ALGORITHM_BUFFER])(self.algorithm_buffers)

        if self.counter >= self.patience:
            self.counter = 0
            self.count_cuts += 1

            with open(self.path, "rb") as source:
                self.main_loop.model.set_parameter_values(
                    load_parameters(source))
                self.main_loop.log.current_row[LOADED_FROM] = self.path

            # Reset algorithm buffer
            for var in self.algorithm_buffers:
                var_val = var.get_value()
                var.set_value(numpy.zeros(var_val.shape, dtype=var_val.dtype))

            self.lr.set_value(float(self.cut_size * self.lr.get_value()))

            if self.count_cuts >= self.num_cuts:
                self.main_loop.log.current_row[
                    'training_finish_requested'] = True
