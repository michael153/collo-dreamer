import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

# import base_agent
import planning_agent
from utils import models, tools, wrappers
from planners import gn_solver

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  print("In load dataset: ", config.train_steps, config.batch_length, config.dataset_balance)
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
    if prefix == 'test':
      tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


class Dreamer(planning_agent.PlanningAgent):

  def __init__(self, config, datadir, actspace, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
    # self._strategy = tf.distribute.MirroredStrategy()
    # with self._strategy.scope():
    #   self._dataset = iter(self._strategy.experimental_distribute_dataset(
    #       load_dataset(datadir, self._c)))
    #   self._build_model()
    self._strategy = None
    self._dataset = iter(load_dataset(datadir, self._c))
    self._build_model()

  @tf.function()
  def train(self, data, log_images=False):
    # self._strategy.experimental_run_v2(self._train, args=(data, log_images))
    self._train(data, log_images)

  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)
      model_loss = self._c.kl_scale * div - sum(likes.values())
      # model_loss /= float(self._strategy.num_replicas_in_sync)

    print("(in _train) Post: ")
    for k in post:
      print(f"(in _train) Post ({k}): {post[k].shape}")

    print("(in _train) self._dynamics.get_feat(post) shape: ", feat.shape)
    print("(in _train) data['image'] shape: ", data['image'].shape)
    print("(in _train) data['reward'] shape: ", data['reward'].shape)
    print("(in _train) data['action'] shape: ", data['action'].shape)

    num_samples = data['image'].shape[1]
    actor_loss = None

    for sample in range(num_samples):
      data_slice = {
        'image': data['image'][:, sample, :, :, :],
        'reward': data['reward'][:, sample],
        'action': data['action'][:, sample, :]
      }
      feat, _ = self.get_init_feat(data_slice, None)
      feat = feat.astype(np.float32)
      print("(in _train) Feature shape: ", feat.shape)
      act_pred, img_pred, feat_pred, info = self._plan(None, False, None, feat, verbose=False)
      if actor_loss is None:
        actor_loss = info['metrics'].action_violation
      else:
        actor_loss += info['metrics'].action_violation

    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
      # value_loss /= float(self._strategy.num_replicas_in_sync)

    model_norm = self._model_opt(model_tape, model_loss)
    # actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    

    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    # self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def pair_residual_func_body(self, x_a, x_b, lam, nu):
    """ This function is required by the gn_solver. It taking in the pair of adjacent states (current and next state)
    and outputs the residuals for the Gauss-Newton optimization
  
    :param x_a: current states and actions, batched across sequence length
    :param x_b: next states and actions
    :param lam: lagrange multiplier for dynamics
    :param nu: lagrange multiplier for actions
    :return: a vector of residuals
    """
    # Compute residuals
    actions_a = x_a[:, -self._actdim:][None]
    feats_a = x_a[:, :-self._actdim][None]
    states_a = self._dynamics.from_feat(feats_a)
    prior_a = self._dynamics.img_step(states_a, actions_a)
    x_b_pred = self._dynamics.get_mean_feat(prior_a)[0]
    dyn_residual = x_b[:, :-self._actdim] - x_b_pred
    act_residual = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    rew = self._reward(x_b[:, :-self._actdim]).mode()[:, None]
    rew_residual = tf.math.softplus(-rew)

    # Compute coefficients
    dyn_c = tf.sqrt(lam)[:, :, None] * self._c.dyn_loss_scale
    act_c = tf.sqrt(nu)[:, :, None] * self._c.act_loss_scale
    rew_c = tf.ones(lam.shape, np.float32)[:, :, None]

    # Normalize with the sum of multipliers to scale the objective in a reasonable range.
    bs, n = nu.shape[0:2]
    normalize = 1 / (tf.reduce_mean(dyn_c, 1) + tf.reduce_mean(act_c, 1) + tf.reduce_mean(rew_c, 1))
    dyn_resw = dyn_c * tf.reshape(dyn_residual, (bs, n, -1))
    act_resw = act_c * tf.reshape(act_residual, (bs, n, -1))
    rew_resw = rew_c * tf.reshape(rew_residual, (bs, n, -1))
    objective = normalize[:, :, None] * tf.concat([dyn_resw, act_resw, rew_resw], 2)

    return tf.reshape(objective, (-1, objective.shape[2]))

  @tf.function
  def opt_step(self, plan, init_feat, lam, nu):
    """ One optimization step. This function is needed for the code to compile properly """
    # We actually also optimize the first state, ensuring it is close to the true first state
    init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
    pair_residual_func = lambda x_a, x_b : self.pair_residual_func_body(x_a, x_b, lam, nu)
    plan = gn_solver.solve_step(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def _plan(self, init_obs, save_images, step, init_feat=None, verbose=True):
    """ The LatCo agent. This function implements the dual descent algorithm. _batch_ optimization procedures are
    executed in parallel, and the best solution is taken.
    
    :param init_obs: Initial observation (either observation of latent has to be specified)
    :param save_images: Whether to save images
    :param step: Index to label the saved images with
    :param init_feat: Initial latent state (either observation of latent has to be specified)
    """
    hor = self._c.horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    batch = self._c.batch_size
    dyn_threshold = self._c.dyn_threshold
    act_threshold = self._c.act_threshold

    if init_feat is None:
      init_feat, _ = self.get_init_feat(init_obs)
    plan = tf.random.normal((batch, (hor + 1) * var_len_step,), dtype=self._float)

    print("(in _plan) Plan shape: ", plan.shape)

    # Set the first state to be the observed initial state
    plan = tf.concat([tf.repeat(init_feat, batch, 0), plan[:, feat_size:]], 1)
    plan = tf.reshape(plan, [batch, hor + 1, var_len_step])
    lam = tf.ones((batch, hor)) * self._c.init_lam
    nu = tf.ones((batch, hor)) * self._c.init_nu

    print("(in _plan) (After concat) Plan shape: ", plan.shape)
    plan = plan.astype(np.float32)

    print("(in _plan) Plan type: ", type(plan))

    # Run dual descent
    plans = [plan]
    metrics = tools.AttrDefaultDict(list)
    for i in range(self._c.optimization_steps):
      # Run Gauss-Newton step
      plan = self.opt_step(plan, init_feat, lam, nu)
      plan_res = tf.reshape(plan, [batch, hor+1, -1])
      feat_preds, act_preds = tf.split(plan_res, [feat_size, self._actdim], 2)
      states = self._dynamics.from_feat(feat_preds[:, :-1])
      priors = self._dynamics.img_step(states, act_preds[:, :-1])
      priors_feat = tf.squeeze(self._dynamics.get_mean_feat(priors))
      dyn_viol = tf.reduce_sum(tf.square(priors_feat - feat_preds[:, 1:]), 2)
      act_viol = tf.reduce_sum(tf.clip_by_value(tf.square(act_preds[:, :-1]) - 1, 0, np.inf), 2)

      # Update lagrange multipliers
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lam_delta = lam * 0.1 * tf.math.log((dyn_viol + 0.1 * dyn_threshold) / dyn_threshold) / tf.math.log(10.0)
        nu_delta  = nu * 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        lam = lam + lam_delta
        nu = nu + nu_delta

      # Logging
      act_preds_clipped = tf.clip_by_value(act_preds, -1, 1)
      metrics.dynamics.append(tf.reduce_sum(dyn_viol))
      metrics.action_violation.append(tf.reduce_sum(act_viol))
      metrics.dynamics_coeff.append(self._c.dyn_loss_scale**2 * tf.reduce_sum(lam))
      metrics.action_coeff.append(self._c.act_loss_scale**2 * tf.reduce_sum(nu))
      plans.append(plan)

      if self._c.log_colloc_scalars:
        # Compute and record dynamics loss and reward
        rew_raw = self._reward(feat_preds).mode()
        metrics.rewards.append(tf.reduce_sum(rew_raw, 1))

        # Record model rewards
        model_feats = self._dynamics.imagine_feat(act_preds_clipped[0:1], init_feat, deterministic=True)
        model_rew = self._reward(model_feats[0:1]).mode()
        metrics.model_rewards.append(tf.reduce_sum(model_rew))

    # Select best plan
    model_feats = self._dynamics.imagine_feat(act_preds_clipped, tf.repeat(init_feat, batch, 0), deterministic=False)
    model_rew = tf.reduce_sum(self._reward(model_feats).mode(), [1])
    best_plan = tf.argmax(model_rew)
    predicted_rewards = model_rew[best_plan]
    metrics.predicted_rewards.append(predicted_rewards)

    # Get action and feature predictions
    act_preds = act_preds[best_plan, :min(hor, self._c.mpc_steps)]
    if tf.reduce_any(tf.math.is_nan(act_preds)) or tf.reduce_any(tf.math.is_inf(act_preds)):
      act_preds = tf.zeros_like(act_preds)
    feat_preds = feat_preds[best_plan, :min(hor, self._c.mpc_steps)]
    if self._c.log_colloc_scalars:
      metrics.rewards = [r[best_plan] for r in metrics.rewards]
    else:
      metrics.rewards = [tf.reduce_sum(self._reward(feat_preds).mode())]

    # Logging
    img_preds = None
    if save_images:
      img_preds = self._decode(feat_preds).mode()
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      self.visualize_colloc(img_preds, act_preds, init_feat, step)
    if verbose:
      if batch > 1:
        print(f'plan rewards: {model_rew}, best plan: {best_plan}')
      print(f"Planned average dynamics loss: {metrics.dynamics[-1] / hor}")
      print(f"Planned average action violation: {metrics.action_violation[-1] / hor}")
      print(f"Planned total reward: {metrics.predicted_rewards[-1] / hor}")
    info = {'metrics': tools.map_dict(lambda x: x[-1] / hor if len(x) > 0 else 0, dict(metrics)),
            'plans': tf.stack(plans, 0)[:, best_plan:best_plan + 1],
            'curves': dict(metrics)}
    return act_preds, img_preds, feat_preds, info

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, model_norm, value_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())

  def _image_summaries(self, data, embed, image_pred):
    truth = data['image'][:6] + 0.5
    recon = image_pred.mode()[:6]
    init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    init = {k: v[:, -1] for k, v in init.items()}
    prior = self._dynamics.imagine(data['action'][:6, 5:], init)
    openl = self._decode(self._dynamics.get_feat(prior)).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    openl = tf.concat([truth, model, error], 2)
    tools.graph_summary(
        self._writer, tools.video_summary, 'agent/openl', openl)

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()
