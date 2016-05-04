import os
from blocks import initialization
from blocks.algorithms import (
    Adam, CompositeRule, GradientDescent, StepClipping)
from blocks.extensions import (Printing, Timing)
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
import cPickle
from blizzard import blizzard_stream
from theano import function
from utils import train_parse

parser = train_parse()
args = parser.parse_args()

exp_name = args.experiment_name
save_dir = args.save_dir

print "Saving config ..."
with open(os.path.join(save_dir, 'config', exp_name + '.pkl'), 'w') as f:
    cPickle.dump(args, f)
print "Finished saving."

w_init = initialization.IsotropicGaussian(0.01)
b_init = initialization.Constant(0.)

train_stream = blizzard_stream(
    ('train',), args.batch_size, args.seq_length)

valid_stream = blizzard_stream(
    ('valid',), args.batch_size, args.seq_length)

f0_tr, f0_mask_tr, spectrum_tr, transcripts_tr, transcripts_mask_tr, start_flag_tr, voiced_tr = next(train_stream.get_epoch_iterator())

if args.model == "simple":
    from model import SimpleParrot as Parrot

    parrot_args = {
        'num_freq': args.num_freq,
        'k': args.num_mixture,
        'k_f0': args.k_f0,
        'rnn1_h_dim': args.rnn1_size,
        'att_size': args.size_attention,
        'num_letters': args.num_letters,
        'sampling_bias': 0.,
        'weights_init': w_init,
        'biases_init': b_init,
        'name': 'parrot'}
else:
    from model import Parrot

    parrot_args = {
        'num_freq': args.num_freq,
        'k': args.num_mixture,
        'k_f0': args.k_f0,
        'rnn1_h_dim': args.rnn1_size,
        'rnn2_h_dim': args.rnn2_size,
        'att_size': args.size_attention,
        'num_letters': args.num_letters,
        'sampling_bias': 0.,
        'weights_init': w_init,
        'biases_init': b_init,
        'name': 'parrot'}

parrot = Parrot(**parrot_args)
parrot.initialize()

f0, f0_mask, voiced, spectrum, transcripts, transcripts_mask, start_flag = \
    parrot.symbolic_input_variables()

cost, extra_updates = parrot.compute_cost(
    f0, f0_mask, voiced, spectrum, transcripts, transcripts_mask,
    start_flag, args.batch_size, args.seq_length)

cost.name = 'nll'
# print function([
#     f0, f0_mask, voiced, spectrum,
#     transcripts, transcripts_mask, start_flag],
#     cost, updates=extra_updates, on_unused_input='warn')(
#     f0_tr, f0_mask_tr, voiced_tr, spectrum_tr,
#     transcripts_tr, transcripts_mask_tr, start_flag_tr)

# import ipdb; ipdb.set_trace()

cg = ComputationGraph(cost)
model = Model(cost)
parameters = cg.parameters

algorithm = GradientDescent(
    cost=cost,
    parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.), Adam(args.learning_rate)]),
    on_unused_sources='warn')
algorithm.add_updates(extra_updates)

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches=args.save_every,
    prefix="train")

valid_monitor = DataStreamMonitoring(
    [cost],
    valid_stream,
    every_n_batches=args.save_every,
    prefix="valid")

extensions = [
    Timing(every_n_batches=args.save_every),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches=args.save_every),
    Checkpoint(
        save_dir + "pkl/best_" + exp_name + ".tar",
        save_separately=['log'],
        use_cpickle=True,
        save_main_loop=False)
    .add_condition(
        ["after_batch"],
        predicate=OnLogRecord('valid_nll_best_so_far')),
    Printing(every_n_batches=args.save_every)]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=extensions)

main_loop.run()
