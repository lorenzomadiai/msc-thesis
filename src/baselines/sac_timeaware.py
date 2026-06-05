# Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
#!/usr/bin/env python

from functools import partial
import numpy as np
import tensorflow as tf
import gym
import time
from wc_sac.utils.logx import EpochLogger
from wc_sac.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from wc_sac.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs
from safety_gym.envs.engine import Engine
from gym.envs.registration import register, registry

# === ADDED ===
from wrappers import TimeBudgetWrapper

EPS = 1e-8

def safe_register(id, **kwargs):
    if id in registry.env_specs:
        return
    register(id=id, **kwargs)

# -------------------- ENVS --------------------
config1 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'goal_locations': [(1.1, 1.1)],
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'hazards_size': 0.7,
    'hazards_keepout': 0.705,
    'hazards_locations': [(0, 0)]
}

safe_register(
    id='StaticEnv-v0',
    entry_point='safety_gym.envs.mujoco:Engine',
    kwargs={'config': config1},
)

config2 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 3,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305
}

safe_register(
    id='DynamicEnv-v0',
    entry_point='safety_gym.envs.mujoco:Engine',
    kwargs={'config': config2}
)

config3 = {
    'placements_extents': [-3.5, -3.5, 3.5, 3.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    "robot_locations": [(-2.5, -2.5)],
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'goal_locations': [(2.5, 2.5)],
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 5,
    'lidar_num_bins': 16,
    'hazards_num': 7,
    'hazards_size': 0.45,
    'hazards_keepout': 0.505,
}

safe_register(
    id='MyThesisStaticEnv-v0',
    entry_point='safety_gym.envs.mujoco:Engine',
    kwargs={'config': config3},
)

config4 = {
    'placements_extents': [-2.5, -2.5, 2.5, 2.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'continue_goal': False,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 5,
    'lidar_num_bins': 16,
    'hazards_num': 7,
    'hazards_size': 0.45,
    'hazards_keepout': 0.505,
}

safe_register(
    id='MyThesisDynamicEnv-v0',
    entry_point='safety_gym.envs.mujoco:Engine',
    kwargs={'config': config4},
)

# -------------------- TF UTILS --------------------
def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(
            targ_vars[v_targ],
            polyak * targ_vars[v_targ] + (1 - polyak) * main_vars[v_main]
        )
        assign_ops.append(assign_op)
    return tf.group(assign_ops)

# -------------------- POLICIES --------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi

# -------------------- ACTORS & CRITICS --------------------
def mlp_actor(x, a, name='pi', hidden_sizes=(64, 64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale
    return mu, pi, logp_pi

def mlp_critic(x, a, pi, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda z: tf.squeeze(
        mlp(x=z,
            hidden_sizes=list(hidden_sizes) + [1],
            activation=activation,
            output_activation=None),
        axis=1
    )

    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x, a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x, pi], axis=-1))

    return critic, critic_pi

# -------------------- REPLAY BUFFER (reward-only) --------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

# -------------------- SAC (reward-only) --------------------
def sac(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-4, batch_size=1024, local_start_steps=int(1e3),
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10, local_update_after=int(1e3),
        update_freq=1, render=False,
        fixed_entropy_bonus=None, entropy_constraint=-1.0,
        reward_scale=1,

        # === time wrapper params ===
        budget_min=None, budget_max=None, deadline_penalty=0.0, eval_max_budget=None,
        ):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    ac_kwargs['action_space'] = env.action_space

    # Inputs (NO cost placeholder)
    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)

    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)

    # Entropy bonus / temperature
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha', initializer=0.0, trainable=True, dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(tf.clip_by_value(alpha, 1e-8, 1e8))

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    if proc_id() == 0:
        var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/qr1', 'main/qr2', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t total: %d\n') % var_counts)

    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)

    # Reward-only backups
    q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * (min_q_pi_targ - alpha * logp_pi2))

    # Losses (NO cost term)
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1) ** 2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2) ** 2)
    q_loss = qr1_loss + qr2_loss

    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    alpha_loss = -alpha * (entropy_constraint - pi_entropy)
    print('using entropy constraint', entropy_constraint)

    train_pi_op = MpiAdamOptimizer(learning_rate=lr).minimize(
        pi_loss, var_list=get_vars('main/pi'), name='train_pi'
    )

    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr).minimize(
            q_loss, var_list=get_vars('main/q'), name='train_q'
        )

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    target_update = get_target_update('main', 'target', polyak)

    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])

    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    sess.run(sync_all_params())

    logger.setup_tf_saver(
        sess,
        inputs={'x': x_ph, 'a': a_ph},
        outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2}
    )

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    # wrapper controls termination => while not d
    def test_agent(n=10):
        for j in range(n):
            o, r, d = test_env.reset(), 0, False
            ep_ret, ep_len, ep_goals = 0, 0, 0
            while not d:
                o, r, d, info = test_env.step(get_action(o, True))
                if render and proc_id() == 0 and j == 0:
                    test_env.render()
                ep_ret += r
                ep_len += 1
                ep_goals += 1 if info.get('goal_met', False) else 0
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len, TestEpGoals=ep_goals)

    start_time = time.time()
    o, r, d = env.reset(), 0, False
    ep_ret, ep_len, ep_goals = 0, 0, 0
    total_steps = steps_per_epoch * epochs

    vars_to_get = dict(
        LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss,
        QR1Vals=qr1, QR2Vals=qr2, LogPi=logp_pi, PiEntropy=pi_entropy,
        Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss
    )

    print('starting training', proc_id())

    number_model = 0
    local_steps = 0
    local_steps_per_epoch = steps_per_epoch // num_procs()
    local_batch_size = batch_size // num_procs()
    epoch_start_time = time.time()

    for t in range(total_steps // num_procs()):
        if t > local_start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, info = env.step(a)
        r *= reward_scale
        cost = info.get('cost', 0)

        r -= 0.02 * cost

        ep_ret += r
        ep_len += 1
        ep_goals += 1 if info.get('goal_met', False) else 0
        local_steps += 1

        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpGoals=ep_goals)
            o, r, d = env.reset(), 0, False
            ep_ret, ep_len, ep_goals = 0, 0, 0

        if t > 0 and t % update_freq == 0:
            for j in range(update_freq):
                batch = replay_buffer.sample_batch(local_batch_size)
                feed_dict = {
                    x_ph: batch['obs1'],
                    x2_ph: batch['obs2'],
                    a_ph: batch['acts'],
                    r_ph: batch['rews'],
                    d_ph: batch['done']
                }
                if t < local_update_after:
                    logger.store(**sess.run(vars_to_get, feed_dict))
                else:
                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)

        if t > 0 and t % local_steps_per_epoch == 0:
            epoch = t // local_steps_per_epoch

            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, number_model)
                number_model += 1

            test_start_time = time.time()
            test_agent()
            logger.store(TestTime=time.time() - test_start_time)

            logger.store(EpochTime=time.time() - epoch_start_time)
            epoch_start_time = time.time()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EpGoals', average_only=True)
            logger.log_tabular('TestEpGoals', average_only=True)
            logger.log_tabular('TotalEnvInteracts', mpi_sum(local_steps))
            logger.log_tabular('QR1Vals', with_min_and_max=True)
            logger.log_tabular('QR2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQR1', average_only=True)
            logger.log_tabular('LossQR2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogAlpha', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TestTime', average_only=True)
            logger.log_tabular('EpochTime', average_only=True)
            logger.log_tabular('TotalTime', time.time() - start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--local_start_steps', default=500, type=int)
    parser.add_argument('--local_update_after', default=500, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    # entropy (keep)
    parser.add_argument('--fixed_entropy_bonus', default=None, type=float)
    parser.add_argument('--entropy_constraint', type=float, default=-1)

    parser.add_argument('--logger_kwargs_str', type=json.loads, default='{"output_dir": "./data"}')

    # time wrapper
    parser.add_argument('--use_time_wrapper', action='store_true')
    parser.add_argument('--budget_min', type=int, default=120)
    parser.add_argument('--budget_max', type=int, default=220)
    parser.add_argument('--deadline_penalty', type=float, default=1.0)
    parser.add_argument('--eval_max_budget', type=int, default=None)

    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)

    from wc_sac.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = args.logger_kwargs_str

    def env_fn():
        base = gym.make(args.env)
        if args.use_time_wrapper:
            return TimeBudgetWrapper(
                base,
                budget_min=args.budget_min,
                budget_max=args.budget_max,
                deadline_penalty=args.deadline_penalty,
                eval_mode=False,
                eval_max_budget=args.eval_max_budget
            )
        return base

    sac(
        env_fn,
        actor_fn=mlp_actor,
        critic_fn=mlp_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        logger_kwargs=logger_kwargs,
        steps_per_epoch=args.steps_per_epoch,
        update_freq=args.update_freq,
        lr=args.lr,
        render=args.render,
        local_start_steps=args.local_start_steps,
        local_update_after=args.local_update_after,
        fixed_entropy_bonus=args.fixed_entropy_bonus,
        entropy_constraint=args.entropy_constraint,
        budget_min=args.budget_min if args.use_time_wrapper else None,
        budget_max=args.budget_max if args.use_time_wrapper else None,
        deadline_penalty=args.deadline_penalty,
        eval_max_budget=args.eval_max_budget
    )
