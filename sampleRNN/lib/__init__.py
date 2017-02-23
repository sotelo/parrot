import ops
#import lasagne
#from theano.compile.nanguardmode import NanGuardMode

import math
import time
import locale

import numpy
import theano
import theano.tensor as T
import theano.gof

import cPickle as pickle
#import pickle
import warnings
import sys, os, errno, glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: Grouping is not working on cluster! :-?
# Set a locale first or you won't get grouping at all
locale.setlocale(locale.LC_ALL, '')
# 'en_US.UTF-8'

_params = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `theano.shared` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `theano.shared`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = theano.shared(*args, **kwargs)
        param.param = True
        _params[name] = param
    return _params[name]

def delete_params(name):
    to_delete = [p_name for p_name in _params if name in p_name]
    for p_name in to_delete:
        del _params[p_name]

def search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function. When optimizing a cost function, you
    can use this to get a list of all of the trainable params in the graph, like
    so:

    `lib.search(cost, lambda x: hasattr(x, "param"))`
    or
    `lib.search(cost, lambda x: hasattr(x, "param") and x.param==True)`
    """

    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, T.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results

    return _search(node, critereon, set())

def floatX(x):
    """
    Convert `x` to the numpy type specified in `theano.config.floatX`.
    """
    if theano.config.floatX == 'float16':
        return numpy.float16(x)
    elif theano.config.floatX == 'float32':
        return numpy.float32(x)
    else: # Theano's default float type is float64
        print "Warning: lib.floatX using float64"
        return numpy.float64(x)

def save_params(path):
    param_vals = {}
    for name, param in _params.iteritems():
        param_vals[name] = param.get_value()

    with open(path, 'wb') as f:
        pickle.dump(param_vals, f)

def load_params(path):
    with open(path, 'rb') as f:
        param_vals = pickle.load(f)

    for name, val in param_vals.iteritems():
        _params[name].set_value(val)

def clear_all_params():
    to_delete = [p_name for p_name in _params]
    for p_name in to_delete:
        del _params[p_name]

def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise

__model_setting_file_name = 'model_settings.txt'
def print_model_settings(locals_var, path=None, sys_arg=False):
    """
    Prints all variables in upper case in locals_var,
    except for T which usually stands for theano.tensor.
    If locals() passed as input to this method, will print
    all the variables in upper case defined so far, that is
    model settings.

    With `path` as an address to a directory it will _append_ it
    as a file named `model_settings.txt` as well.

    With `sys_arg` set to True, log information about Python, Numpy,
    and Theano and passed arguments to the script will be added too.
    args.pkl would be overwritten, specially in case of resuming a job.
    But again that wouldn't be much of a problem as all the passed args
    to the script except for '--resume' should be the same.

    With both `path` and `sys_arg` passed, dumps the theano.config.

    :usage:
        >>> import theano.tensor as T
        >>> import lib
        >>> BATCH_SIZE, DIM = 128, 512
        >>> DATA_PATH = '/Path/to/dataset'
        >>> lib.print_model_settings(locals(), path='./')
    """
    log = ""
    if sys_arg:
        try:
            log += "Python:\n"
            log += "\tsys.version_info\t{}\n".format(str(sys.version_info))
            log += "Numpy:\n"
            log += "\t.__version__\t{}\n".format(numpy.__version__)
            log += "Theano:\n"
            log += "\t.__version__\t{}\n".format(theano.__version__)
            log += "\n\nAll passed args:\n"
            log += str(sys.argv)
            log += "\n"
        except:
            print "Something went wrong during sys_arg logging. Continue anyway!"

    log += "\nModel settings:"
    all_vars = [(k,v) for (k,v) in locals_var.items() if (k.isupper() and k != 'T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        log += ("\n\t%-20s %s" % (var_name, var_value))
    print log
    if path is not None:
        ensure_dir(path)
        # Don't override, just append if by mistake there is something in the file.
        with open(os.path.join(path, __model_setting_file_name), 'a+') as f:
            f.write(log)
        if sys_arg:
            with open(os.path.join(path, 'th_conf.txt'), 'a+') as f:
                f.write(str(theano.config))
            with open(os.path.join(path, 'args.pkl'), 'wb') as f:
                pickle.dump(sys.argv, f)
                # To load:
                # >>> import cPickle as pickle
                # >>> args = pickle.load(open(os.path.join(path, 'args.pkl'), 'rb'))

def get_params(cost, criterion=lambda x: hasattr(x, 'param') and x.param==True):
    """
    Default criterion:
        lambda x: hasattr(x, 'param') and x.param==True
    This will return every parameter for cost from computation graph.

    To exclude a parameter, just set 'param' to False:
        >>> h0 = lib.param('h0',\
                numpy.zeros((3, 2*512), dtype=theano.config.floatX))
        >>> print h0.param  # Default: True
        >>> h0.param = False

    In this case one still can get list of all params (False or True) by:
        >>> lib.get_params(cost, lambda x: hasattr(x, 'param')

    :returns:
        A list of params
    """
    return search(cost, criterion)

def print_params_info(params, path=None):
    """
    Print information about the parameters in the given param set.

    With `path` as an address to a directory it will _append_ it
    as a file named `model_settings.txt` as well.

    :usage:
        >>> params = lib.get_params(cost)
        >>> lib.print_params_info(params, path='./')
    """
    params = sorted(params, key=lambda p: p.name)
    values = [p.get_value(borrow=True) for p in params]
    shapes = [p.shape for p in values]
    total_param_count = 0
    multiply_all = lambda a, b: a*b
    log = "\nParams for cost:"
    for param, value, shape in zip(params, values, shapes):
        log += ("\n\t%-20s %s" % (shape, param.name))
        total_param_count += reduce(multiply_all, shape)

    log += "\nTotal parameter count for this cost:\n\t{0}".format(
        locale.format("%d", total_param_count, grouping=True)
    )
    print log

    if path is not None:
        ensure_dir(path)
        # Don't override, just append if by mistake there is something in the file.
        with open(os.path.join(path, __model_setting_file_name), 'a+') as f:
            f.write(log)

__train_log_file_name = 'train_log.pkl'
def save_training_info(values, path):
    """
    Gets a set of values as dictionary and append them to a log file.
    stores in <path>/train_log.pkl
    """
    file_name = os.path.join(path, __train_log_file_name)
    try:
        with open(file_name, "rb") as f:
            log = pickle.load(f)
    except IOError:  # first time
        log = {}
        for k in values.keys():
            log[k] = []
    for k, v in values.items():
        log[k].append(v)
    with open(file_name, "wb") as f:
        pickle.dump(log, f)

resume_key = 'last resume index'
def resumable(path,
              iter_key='iter',
              epoch_key='epoch',
              add_resume_counter=True,
              other_keys=[]):
    """
    :warning:
        This is a naive implementation of resuming a training session
        and does not save and reload the training loop. The serialization
        of training loop and everything is costly and error-prone.

    :todo:
        - Save and load a serializable training loop. (See warning above)
        - Heavily dependent on the "model" file and the names used there right
          now. It's really easy to miss anything.

    `path` should be pointing at the root directory where `train_log.pkl`
    (See __train_log_file_name) and `params/` reside.

    Always assuming all the values in the log dictionary (except `resume_key`),
    are lists with the same length.
    """
    file_name = os.path.join(path, __train_log_file_name)
    # Raise error if does not exists.
    with open(file_name, "rb") as f:
        log = pickle.load(f)

    param_found = False
    res_path = os.path.join(path, 'params', 'params_e{}_i{}*.pkl')
    for reverse_idx in range(-1, -len(log[epoch_key])-1, -1):
        ep, it = log[epoch_key][reverse_idx], log[iter_key][reverse_idx]
        print "> Params file for epoch {} iter {}".format(ep, it),
        last_path = glob.glob(res_path.format(ep, it))
        if len(last_path) == 1:
            res_path = last_path[0]
            param_found = True
            print "found."
            break
        elif len(last_path) == 0:
            print "[NOT FOUND]. FALLING BACK TO..."
        else:  # > 1
            # choose one, warning, rare
            print "[multiple version found]:"
            for l_path in last_path:
                print l_path
            res_path = last_path[0]
            param_found = True
            print "Arbitrarily choosing first:\n\t{}".format(res_path)

    assert 'reverse_idx' in locals(), 'Empty train_log???\n{}'.format(log)
    # Finishing for loop with no success
    assert param_found, 'No matching params file with train_log'

    acceptable_len = reverse_idx+len(log[epoch_key])+1
    if acceptable_len != len(log[epoch_key]):
        # Backup of the old train_log
        with open(file_name+'.backup', 'wb') as f:
            pickle.dump(log, f)

        # Change the log file to match the last existing checkpoint.
        for k, v in log.items():
            # Fix resume indices
            if k == resume_key:
                log[k] = [i for i in log[k] if i < acceptable_len]
                continue
            # Rest is useless with no param file.
            log[k] = v[:acceptable_len]

    epochs = log[epoch_key]
    iters = log[iter_key]

    if add_resume_counter:
        resume_val = len(epochs)
        if not resume_key in log.keys():
            log[resume_key] = [resume_val]
        else:
            if log[resume_key] == [] or log[resume_key][-1] != resume_val:
                log[resume_key].append(resume_val)
        with open(file_name, "wb") as f:
            pickle.dump(log, f)

    last_epoch = epochs[-1]
    last_iter = iters[-1]

    # The if-else statement is more readable than `next`:
    #iters_to_consume = next((last_iter%(i-1) for (e, i) in\
    #       zip(epochs, iters) if e == 1), last_iter)
    if last_epoch == 0:
        iters_to_consume = last_iter
    else:
        for e, i in zip(epochs, iters):
            # first time. Epoch turns from 0 to 1.
            # At the end of each `epoch` there should be
            # a monitoring step so it will gives number
            # number of iterations per epoch
            if e == 1:
                iters_per_epoch = i - 1
                break
        iters_to_consume = last_iter % iters_per_epoch

    last_other_keys = [log[k][-1] for k in other_keys]
    return iters_to_consume, res_path, last_epoch, last_iter, last_other_keys

def plot_traing_info(x, ylist, path):
    """
    Loads log file and plot x and y values as provided by input.
    Saves as <path>/train_log.png
    """
    file_name = os.path.join(path, __train_log_file_name)
    try:
        with open(file_name, "rb") as f:
            log = pickle.load(f)
    except IOError:  # first time
        warnings.warn("There is no {} file here!!!".format(file_name))
        return
    plt.figure()
    x_vals = log[x]
    for y in ylist:
        y_vals = log[y]
        if len(y_vals) != len(x_vals):
            warning.warn("One of y's: {} does not have the same length as x:{}".format(y, x))
        plt.plot(x_vals, y_vals, label=y)
        # assert len(y_vals) == len(x_vals), "not the same len"
    plt.xlabel(x)
    plt.legend()
    #plt.show()
    plt.savefig(file_name[:-3]+'png', bbox_inches='tight')
    plt.close('all')

def create_logging_folders(path):
    """
    Handle structure of folders and naming here instead of training file.

    :todo:
        - Duh. Implement!
    """
    pass

def tv(var):
    """
    :todo:
        - add tv() function for theano variables so that instead of calling
        x.tag.test_value, you can get the same thing just by calling the method
        in a faster way...
        - also for x.tag.test_value.shape
    """
    # Based on EAFP (easier to ask for forgiveness than permission)
    try:
        return var.tag.test_value
    except AttributeError:
        print "NONE, test_value has not been set."
        import ipdb; ipdb.set_trace()

    ## Rather than LBYL (look before you leap)
    #if hasattr(var, 'tag'):
    #    if hasattr(var.tag, 'test_value'):
    #        return var.tag.test_value
    #   else:
    #       print "NONE, test_value has not set."
    #       import ipdb; ipdb.set_trace()
    #else:
    #    print "NONE, tag has not set."
    #    import ipdb; ipdb.set_trace()

def tvs(var):
    """
    :returns:
        var.tag.test_value.shape
    """
    return tv(var).shape

def _is_symbolic(v):
    r"""Return `True` if any of the arguments are symbolic.
    See:
        https://github.com/Theano/Theano/wiki/Cookbook
    """
    symbolic = False
    v = list(v)
    for _container, _iter in [(v, xrange(len(v)))]:
        for _k in _iter:
            _v = _container[_k]
            if isinstance(_v, theano.gof.Variable):
                symbolic = True
    return symbolic

def unique_list(inp_list):
    """
    returns a list with unique values of inp_list.
    :usage:
        >>> inp_list = ['a', 'b', 'c']
        >>> unique_inp_list = unique_list(inp_list*2)
    """
    return list(set(inp_list))
