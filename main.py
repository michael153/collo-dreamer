# import argparse
# import collections
# import functools
# import json
# import os
# import pathlib
# import sys
# import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['MUJOCO_GL'] = 'egl'

# import numpy as np
import tensorflow as tf
# from tensorflow.keras.mixed_precision import experimental as prec

# tf.get_logger().setLevel('ERROR')

# from tensorflow_probability import distributions as tfd

# sys.path.append(str(pathlib.Path(__file__).parent))

from utils import models, tools, wrappers
# from dreamer import Dreamer, count_steps, summarize_episode


# def define_config():
#   config = tools.AttrDict()
#   # General.
#   config.logdir = pathlib.Path('.')
#   config.seed = 0
#   config.steps = 5e6
#   config.eval_every = 1e4
#   config.log_every = 1e3
#   config.log_scalars = True
#   config.log_images = True
#   config.gpu_growth = True
#   config.precision = 16
#   # Environment.
#   config.task = 'dmc_walker_walk'
#   config.envs = 1
#   config.parallel = 'none'
#   config.action_repeat = 2
#   config.time_limit = 1000
#   config.prefill = 5000
#   config.eval_noise = 0.0
#   config.clip_rewards = 'none'
#   # Model.
#   config.deter_size = 200
#   config.stoch_size = 30
#   config.num_units = 400
#   config.dense_act = 'elu'
#   config.cnn_act = 'relu'
#   config.cnn_depth = 32
#   config.pcont = False
#   config.free_nats = 3.0
#   config.kl_scale = 1.0
#   config.pcont_scale = 10.0
#   config.weight_decay = 0.0
#   config.weight_decay_pattern = r'.*'
#   # Training.
#   config.batch_size = 1
#   config.batch_length = 15
#   config.train_every = 1000
#   config.train_steps = 100
#   config.pretrain = 100
#   config.model_lr = 6e-4
#   config.value_lr = 8e-5
#   config.actor_lr = 8e-5
#   config.grad_clip = 100.0
#   config.dataset_balance = False
#   # Behavior.
#   config.discount = 0.99
#   config.disclam = 0.95
#   config.horizon = 15
#   config.action_dist = 'tanh_normal'
#   config.action_init_std = 5.0
#   config.expl = 'additive_gaussian'
#   config.expl_amount = 0.3
#   config.expl_decay = 0.0
#   config.expl_min = 0.0
#   # LatCo parameters
#   config.optimization_steps = 200
#   config.n_parallel_plans = 15 # replaced by batch_size
#   config.dyn_loss_scale = 1
#   config.act_loss_scale = 1
#   # Lagrange multipliers
#   config.lm_update_every = 1
#   config.init_lam = 1
#   config.lam_lr = 1
#   config.init_nu = 1
#   config.nu_lr = 100
#   config.dyn_threshold = 1e-4
#   config.act_threshold = 1e-4
#   # GN parameters
#   config.gn_damping = 1e-3
#   config.reward_stats = True
#   return config


# def make_env(config, writer, prefix, datadir, store):
#   suite, task = config.task.split('_', 1)
#   print(suite, task)
#   if suite == 'dmc':
#     env = wrappers.DeepMindControl(task)
#     env = wrappers.ActionRepeat(env, config.action_repeat)
#     env = wrappers.NormalizeActions(env)
#   elif suite == 'atari':
#     env = wrappers.Atari(
#         task, config.action_repeat, (64, 64), grayscale=False,
#         life_done=True, sticky_actions=True)
#     env = wrappers.OneHotAction(env)
#   else:
#     raise NotImplementedError(suite)
#   env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
#   callbacks = []
#   if store:
#     callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
#   callbacks.append(
#       lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
#   env = wrappers.Collect(env, callbacks, config.precision)
#   env = wrappers.RewardObs(env)
#   return env


# def main(config):
#   if config.gpu_growth:
#     for gpu in tf.config.experimental.list_physical_devices('GPU'):
#       tf.config.experimental.set_memory_growth(gpu, True)
#   assert config.precision in (16, 32), config.precision
#   if config.precision == 16:
#     prec.set_policy(prec.Policy('mixed_float16'))
#   config.steps = int(config.steps)
#   config.logdir.mkdir(parents=True, exist_ok=True)
#   print('Logdir', config.logdir)

#   # Create environments.
#   datadir = config.logdir / 'episodes'
#   writer = tf.summary.create_file_writer(
#       str(config.logdir), max_queue=1000, flush_millis=20000)
#   writer.set_as_default()
#   train_envs = [wrappers.Async(lambda: make_env(
#       config, writer, 'train', datadir, store=True), config.parallel)
#       for _ in range(config.envs)]
#   test_envs = [wrappers.Async(lambda: make_env(
#       config, writer, 'test', datadir, store=False), config.parallel)
#       for _ in range(config.envs)]
#   actspace = train_envs[0].action_space

#   # Prefill dataset with random episodes.
#   step = count_steps(datadir, config)
#   prefill = max(0, config.prefill - step)
#   print(f'Prefill dataset with {prefill} steps.')
#   random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
#   tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
#   writer.flush()

#   # Train and regularly evaluate the agent.
#   step = count_steps(datadir, config)
#   print(f'Simulating agent for {config.steps-step} steps.')
#   agent = Dreamer(config, datadir, actspace, writer)
#   print("Initialized agent")
#   if (config.logdir / 'variables.pkl').exists():
#     print('Load checkpoint.')
#     agent.load(config.logdir / 'variables.pkl')
#   state = None
#   while step < config.steps:
#     print('Start evaluation.')
#     tools.simulate(
#         functools.partial(agent, training=False), test_envs, episodes=1)
#     writer.flush()
#     print('Start collection.')
#     steps = config.eval_every // config.action_repeat
#     state = tools.simulate(agent, train_envs, steps, state=state)
#     step = count_steps(datadir, config)
#     agent.save(config.logdir / 'variables.pkl')
#   for env in train_envs + test_envs:
#     env.close()


# if __name__ == '__main__':
#   try:
#     import colored_traceback
#     colored_traceback.add_hook()
#   except ImportError:
#     pass
#   parser = argparse.ArgumentParser()
#   for key, value in define_config().items():
#     parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
#   main(parser.parse_args())


# from utils import models

d = models.DenseDecoder((), 3, 400, act=tf.nn.elu)
print(d.variables)