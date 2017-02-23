import os
from blocks import initialization
from blocks.algorithms import (
    Adam, CompositeRule, GradientDescent, StepClipping)
from blocks.extensions import (Printing, Timing)
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
import cPickle
from extensions import LearningRateSchedule, Plot, TimedFinish
from datasets import parrot_stream
from model import Parrot
from utils import train_parse

args = train_parse()

exp_name = args.experiment_name
save_dir = args.save_dir

print "Saving config ..."
with open(os.path.join(save_dir, 'config', exp_name + '.pkl'), 'w') as f:
    cPickle.dump(args, f)
print "Finished saving."

w_init = initialization.IsotropicGaussian(0.01)
b_init = initialization.Constant(0.)

train_stream = parrot_stream(
    args.dataset, args.use_speaker, ('train',), args.batch_size,
    noise_level=args.feedback_noise_level, labels_type=args.labels_type,
    seq_size=args.seq_size, raw_data=args.raw_output)

if args.feedback_noise_level is None:
    val_noise_level = None
else:
    val_noise_level = 0.

valid_stream = parrot_stream(
    args.dataset, args.use_speaker, ('valid',), args.batch_size,
    noise_level=val_noise_level, labels_type=args.labels_type,
    seq_size=args.seq_size, raw_data=args.raw_output)

example_batch = next(train_stream.get_epoch_iterator())

for idx, source in enumerate(train_stream.sources):
    if source not in ['start_flag', 'feedback_noise_level']:
        print source, "shape: ", example_batch[idx].shape, \
            source, "dtype: ", example_batch[idx].dtype
    else:
        print source, ": ", example_batch[idx]

parrot_args = {
    'input_dim': args.input_dim,
    'output_dim': args.output_dim,
    'rnn_h_dim': args.rnn_h_dim,
    'readouts_dim': args.readouts_dim,
    'weak_feedback': args.weak_feedback,
    'full_feedback': args.full_feedback,
    'feedback_noise_level': args.feedback_noise_level,
    'layer_norm': args.layer_norm,
    'use_speaker': args.use_speaker,
    'num_speakers': args.num_speakers,
    'speaker_dim': args.speaker_dim,
    'which_cost': args.which_cost,
    'num_characters': args.num_characters,
    'attention_type': args.attention_type,
    'attention_alignment': args.attention_alignment,
    'encoder_type': args.encoder_type,
    'weights_init': w_init,
    'biases_init': b_init,
    'raw_output': args.raw_output,
    'name': 'parrot'}

parrot = Parrot(**parrot_args)
parrot.initialize()

features, features_mask, labels, labels_mask, speaker, start_flag, raw_sequence = \
    parrot.symbolic_input_variables()

cost, extra_updates, attention_vars, cost_raw = parrot.compute_cost(
    features, features_mask, labels, labels_mask,
    speaker, start_flag, args.batch_size, raw_audio=raw_sequence)

cost_name = args.which_cost
cost.name = cost_name

if parrot.raw_output:
    cost_raw.name = "sampleRNN_cost"

cg = ComputationGraph(cost)
model = Model(cost)

parameters = cg.parameters

step_rule = CompositeRule(
    [StepClipping(10. * args.grad_clip), Adam(args.learning_rate)])

algorithm = GradientDescent(
    cost=cost,
    parameters=parameters,
    step_rule=step_rule,
    on_unused_sources='warn')
algorithm.add_updates(extra_updates)

monitoring_vars = [cost]
plot_names = [['train_' + cost_name, 'valid_' + cost_name]]

if args.lr_schedule:
    lr = algorithm.step_rule.components[1].learning_rate
    monitoring_vars.append(lr)
    plot_names += [['valid_learning_rate']]

if parrot.raw_output:
    monitoring_vars.append(cost_raw)
    plot_names.append(['train_sampleRNN_cost', 'valid_sampleRNN_cost'])



train_monitor = TrainingDataMonitoring(
    variables=monitoring_vars,
    every_n_batches=args.save_every,
    prefix="train")

valid_monitor = DataStreamMonitoring(
    monitoring_vars,
    valid_stream,
    every_n_batches=args.save_every,
    after_epoch=False,
    prefix="valid")

extensions = []

if args.load_experiment:
    extensions += [Load(os.path.join(
        save_dir, "pkl", "best_" + args.load_experiment + ".tar"))]

extensions += [
    Timing(every_n_batches=args.save_every),
    train_monitor]

extensions += [
    valid_monitor,
    TrackTheBest(
        'valid_' + cost_name,
        every_n_batches=args.save_every,
        before_first_epoch=True),
    Plot(
        os.path.join(save_dir, "progress", exp_name + ".png"),
        plot_names,
        every_n_batches=args.save_every,
        email=False),
    Checkpoint(
        os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
        after_training=False,
        save_separately=['log'],
        use_cpickle=True,
        save_main_loop=False,
        before_first_epoch=True)
    .add_condition(
        ["after_batch", "before_training"],
        predicate=OnLogRecord('valid_'+ cost_name + '_best_so_far')),
    Checkpoint(
        os.path.join(save_dir, "pkl", "last_" + exp_name + ".tar"),
        after_training=True,
        save_separately=['log'],
        use_cpickle=True,
        every_n_batches=args.save_every,
        save_main_loop=False)]

if args.lr_schedule:
    extensions += [
        LearningRateSchedule(
            lr, 'valid_' + cost_name,
            os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
            patience=10,
            num_cuts=5,
            every_n_batches=args.save_every)]


extensions += [
    Printing(
        after_epoch=False,
        every_n_batches=args.save_every)]

if args.time_limit:
    extensions += [TimedFinish(args.time_limit)]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=extensions)

print "Training starting:"
main_loop.run()
