import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import argparse
import cv2
import gym
import logging
import os

from collections import deque

import network

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=True, type=str, help='Training directory')
parser.add_argument('--env_name', required=True, type=str, help='Environment name like Breakout-v0')
parser.add_argument('--max_steps', default=1000000, type=int, help='Maximum number of training steps')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--explore_eps', default=0.99, type=float, help='Initial exploration epsilon in range [0, 1]')

FLAGS = parser.parse_args()

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

def preprocess(config, state):
    input_shape = config.get('input_shape')

    state = state.astype(np.float32)
    state = 0.2126 * state[:, :, 0] + 0.7152 * state[:, :, 1] + 0.0722 * state[:, :, 2]

    state = cv2.resize(state, (input_shape[0], input_shape[1]))
    state = np.reshape(state, (input_shape[0], input_shape[1], 1))

    #mean = np.mean(state)
    #std = np.std(state)
    #adj_std = max(std, 1.0/np.sqrt(np.prod(state.shape)))

    #ret = (state - std) / adj_std

    ret = state / 255.

    return ret

class stacked_env(gym.Wrapper):
    def __init__(self, config, env):
        super(stacked_env, self).__init__(env)

        self.env = env
        self.config = config
        self.max_len = config.get('state_stack_size')
        self.stack = deque(maxlen=self.max_len)

    def append(self, state):
        state = preprocess(self.config, state)

        self.stack.append(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.append(state)

        ns = np.concatenate(self.stack, axis=2)
        return ns, reward, done, info

    def reset(self):
        state = self.env.reset()

        self.stack.clear()
        for _ in range(self.max_len):
            self.append(state)

        return np.concatenate(self.stack, axis=2)


def create_one_network(config, states, scope='dueling', reuse=False):
    out = network.create_network(config, states, scope=scope, reuse=reuse)

    value, advantage = tf.split(out, [1, config.get('num_actions')], axis=1)
    adv_mean = tf.reduce_mean(advantage, axis=1)

    q_val = value + advantage - tf.expand_dims(adv_mean, axis=1)

    return q_val

def create_model(config):
    hvd.init()

    checkpoint_dir = config.get('checkpoint_dir')
    if hvd.rank() != 0:
        checkpoint_dir = None

    input_shape = config.get('input_shape')
    state_stack_size = config.get('state_stack_size')
    input_shape[2] *= state_stack_size

    num_actions = config.get('num_actions')

    states_main_ph = tf.placeholder(tf.float32, [None] + input_shape, name='input_states_main')
    states_follower_ph = tf.placeholder(tf.float32, [None] + input_shape, name='input_states_follower')
    qvals_ph = tf.placeholder(tf.float32, [None, num_actions], name='qvals')
    explore_eps_ph = tf.placeholder(tf.float32, [], name='explore_eps')

    q_val_main = create_one_network(config, states_main_ph, scope='dueling_main')
    q_val_follower = create_one_network(config, states_follower_ph, scope='dueling_follower')

    assign_ops = []
    for v_main, v_follower in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dueling_main'), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dueling_follower')):
        op = tf.assign(v_follower, v_main)
        assign_ops.append(op)

    transfer_op = tf.group(assign_ops)

    q_loss = tf.reduce_mean(tf.square(q_val_main - qvals_ph))
    tf.summary.scalar('q_loss', q_loss)

    total_loss = q_loss

    ru = tf.random.uniform([])
    random_action_op = tf.random.uniform([1], minval=0, maxval=num_actions, dtype=tf.int32)
    predict_action_op = tf.cond(tf.less(ru, explore_eps_ph), lambda: random_action_op, lambda: tf.argmax(q_val_follower, axis=1, output_type=tf.int32))

    global_step_op = tf.train.get_or_create_global_step()
    lr_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate_ph')
    tf.summary.scalar('learning_rate', lr_ph)

    #opt = tf.train.AdamOptimizer(lr_ph, beta1=0.9, beta2=0.999, epsilon=0.01)
    opt = tf.train.RMSPropOptimizer(lr_ph, decay=0.99, name='optimizer')

    train_op = opt.minimize(total_loss, global_step=global_step_op)

    opt = hvd.DistributedOptimizer(opt)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    #hooks = []

    variables_to_restore = []
    variables_to_init = []
    for v in tf.global_variables():
        if False:
            names_to_init = ['transform_', 'first_layer']
            initialized = False
            for ni in names_to_init:
                if ni in v.name:
                    variables_to_init.append(v)
                    initialized = True
                    break

            if initialized:
                continue

        variables_to_restore.append(v)

    init_op = tf.initializers.variables(variables_to_init)

    total_saver = tf.train.Saver()
    saver = tf.train.Saver(var_list=variables_to_restore)

    def init_fn(scaffold, sess):
        sess.run([init_op])

        if checkpoint_dir:
            path = tf.train.latest_checkpoint(checkpoint_dir)

            logging.info('checkpoint_dir: {}, restore path: {}'.format(checkpoint_dir, path))
            if path:
                saver.restore(sess, path)

        logging.info('init_fn has been completed')

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge(summaries)
    if checkpoint_dir:
        summary_writer = tf.summary.FileWriter(checkpoint_dir)

    scaffold = tf.train.Scaffold(saver=None, init_fn=init_fn)

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.train.MonitoredTrainingSession(config=session_config, hooks=hooks, checkpoint_dir=None, scaffold=scaffold) as mon_sess:
        sess = mon_sess._tf_sess()

        step = sess.run(global_step_op)
        if hvd.rank() == 0:
            logging.info('step: {}, max_steps: {}'.format(step, FLAGS.max_steps))

        def periodic_tasks(force, last_feed_dict):
            if force or (step % update_follower_steps == 0):
                _ = sess.run(transfer_op)
                #logging.info('Follower weights have been updated')

            if checkpoint_dir:
                if force or (step % summary_steps == 0):
                    if last_feed_dict:
                        summary_str, gs = sess.run([summary_op, global_step_op], feed_dict=last_feed_dict)
                        summary_writer.add_summary(summary_str, gs)

                if force or (step % checkpoint_steps == 0):
                    total_saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step_op, write_meta_graph=True)
                    logging.info('model has been saved with step {}'.format(sess.run(global_step_op)))

        step_gen = config.get('step_gen')
        reset_gen = config.get('reset_gen')
        batch_size = config.get('batch_size')
        discount_gamma = config.get('discount_gamma')
        update_follower_steps = config.get('update_follower_steps')
        summary_steps = config.get('summary_steps')
        checkpoint_steps = config.get('checkpoint_steps')
        alpha = config.get('alpha')
        exploration_discount = config.get('exploration_discount')
        history_size = config.get('history_size')
        learning_rate_decay_factor = config.get('learning_rate_decay_factor')
        learning_rate_decay_steps = config.get('learning_rate_decay_steps')

        history = []

        explore_eps = config.get('explore_eps')
        initial_lr = config.get('initial_learning_rate')
        minimal_lr = config.get('minimal_learning_rate')
        episode_num = 0
        last_episode_rewards = deque(maxlen=100)
        while step < FLAGS.max_steps:
            if mon_sess.should_stop():
                break

            norm = int(step / learning_rate_decay_steps) + 1
            lr = initial_lr / norm
            if lr < minimal_lr:
                lr = minimal_lr

            done = False
            state = reset_gen()

            episode_reward = 0
            while not done:
                feed_dict_pred = {
                    states_follower_ph: [state],
                    explore_eps_ph: explore_eps,
                }
                action = sess.run(predict_action_op, feed_dict = feed_dict_pred)[0]

                new_state, reward, done = step_gen(action)
                episode_reward += reward

                history.append((state, action, reward, new_state, done))
                if len(history) > history_size:
                    history = history[1:]

                state = new_state

                if len(history) < batch_size:
                    continue

                states = []
                next_states = []

                batch_idx = np.random.choice(len(history), batch_size, replace=False)
                for bidx in batch_idx:
                    b = history[bidx]
                    s, a, r, ns, d = b

                    states.append(s)
                    next_states.append(ns)

                feed_dict_pred = {
                    states_main_ph: states,
                    states_follower_ph: next_states,
                }
                qvals, next_qvals = sess.run([q_val_main, q_val_follower], feed_dict=feed_dict_pred)

                for seq_idx, bidx in enumerate(batch_idx):
                    b = history[bidx]
                    s, a, r, ns, d = b

                    qmax_next = np.amax(next_qvals[seq_idx])
                    if d:
                        qmax_next = 0

                    current_qa = qvals[seq_idx][a]
                    # clip Q value to [-1;+1] interval
                    #qsa = min(1, max(-1, (1. - self.alpha) * current_qa + self.alpha * (r + self.discount_gamma * qmax_next)))
                    qsa = (1. - alpha) * current_qa + alpha * (r + discount_gamma * qmax_next)
                    qvals[seq_idx][a] = qsa

                feed_dict_train = {
                    states_main_ph: states,
                    qvals_ph: qvals,
                    lr_ph: lr,
                }
                ret = sess.run([train_op, total_loss, global_step_op], feed_dict=feed_dict_train)
                step += 1

                _, loss_value, gs = ret
                periodic_tasks(force=False, last_feed_dict=feed_dict_train)

                #logging.info('step: {}/{}, loss: {}'.format(step, gs, loss_value))

            last_episode_rewards.append(episode_reward)
            gs = sess.run(global_step_op)

            if hvd.rank() == 0:
                logging.info('step: {}/{}, episode: {}, reward: {}, exploration: {:.1f}%, lr: {:.2e}, mean reward for {} episodes: {:.1f}'.format(
                    step, gs, episode_num, episode_reward, explore_eps*100, lr, len(last_episode_rewards), np.mean(last_episode_rewards)))

            episode_num += 1
            explore_eps *= exploration_discount
            if explore_eps < 0.01:
                explore_eps = 0.05

        periodic_tasks(force=True, last_feed_dict=None)

def main():
    env = gym.make(FLAGS.env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    num_actions = env.action_space.n

    checkpoint_dir = os.path.join(FLAGS.train_dir, 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(checkpoint_dir, 'train.log'), filemode='a', level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    np.set_printoptions(formatter={'float': '{:0.6f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True)

    config = {
        'state_stack_size': 4,
        'checkpoint_dir': checkpoint_dir,
        'discount_gamma': 0.99,
        'alpha': 1,
        'history_size': 20000,
        'update_follower_steps': 800,
        'summary_steps': 100,
        'checkpoint_steps': 10000,
        'explore_eps': FLAGS.explore_eps,
        'exploration_discount': 0.99,
        'num_dense_units': 1024,
        'num_actions': num_actions,
        'num_outputs': num_actions + 1, # value
        'batch_size': FLAGS.batch_size,
        'input_shape': [84, 84, 1],
        'initial_learning_rate': 2.5e-4,
        'minimal_learning_rate': 1e-5,
        'learning_rate_decay_factor': 0.7,
        'learning_rate_decay_steps': 100000,
    }

    env = stacked_env(config, env)

    def step(action):
        state, reward, done, info = env.step(action)
        return state, reward, done

    def reset():
        state = env.reset()
        return state

    config['step_gen'] = step
    config['reset_gen'] = reset

    create_model(config)

if __name__ == '__main__':
    main()
