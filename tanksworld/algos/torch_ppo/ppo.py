# modified code from spinningup OpenAI
import pprint
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gym
import time
from .utils.logx import EpochLogger
from .utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from .utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import sys
from mpi4py import MPI
from arena5.core.utils import mpi_print
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from . import core
import cv2
import os
import json
from math import ceil, sqrt

from stable_baselines.trpo_mpi.utils import flatten_lists
from algos.torch_ppo.mappo_utils import valuenorm
from algos.torch_ppo.mappo_utils.util import huber_loss

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, use_rnn=False):
        if use_rnn:
            self.obs_buf = np.zeros(core.combined_shape_v2(size, 5, obs_dim), dtype=np.float32)
        else:
            self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, comm):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(comm, self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}


class PPOPolicy():
    def __init__(self, env, policy_comm, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.comm = policy_comm
        self.eval_mode = eval_mode

    def run(self, num_steps, data_dir, policy_record=None):

        # if self.external_saved_file is not None:
        #        self.model = PPO1.load(self.external_saved_file, self.env, self.comm)
        # elif os.path.exists(data_dir+"/ppo_save.pkl") or os.path.exists(data_dir+"/ppo_save.zip"):
        #        self.model = PPO1.load(data_dir+"ppo_save", self.env, self.comm)
        #        print("loaded model from saved file!")

        if self.eval_mode:
            self.evaluate(policy_record, num_steps, **self.kargs)
        else:
            self.kargs.update({'steps_to_run': num_steps // self.comm.Get_size()})
            self.learn(policy_record, **self.kargs)
            # if policy_record is not None:
            #        policy_record.save()
            #        self.model.save(policy_record.data_dir+"ppo_save")

    def evaluate(self, policy_record, total_timesteps, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), **kargs):

        local_steps = int(total_timesteps / self.comm.Get_size())
        steps = 0
        observation = self.env.reset()

        ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        ac.load_state_dict(torch.load(self.kargs['model_path']))
        ac.eval()

        eplen = 0
        epret = 0
        running_reward_mean = 0.0
        running_reward_std = 0.0
        num_dones = 0

        pp = pprint.PrettyPrinter(indent=4)

        while steps < local_steps:
            # observation = np.array(observation).reshape(-1, 72)
            with torch.no_grad():
                action, v, logp = ac.step(torch.as_tensor(observation, dtype=torch.float32).to(device))

            # step environment
            observation, reward, done, info = self.env.step(action)

            steps += 1
            eplen += 1
            epret += reward

            if done:
                observation = self.env.reset()

                episode_reward = self.comm.allgather(epret)
                episode_length = self.comm.allgather(eplen)
                episode_statistics = self.comm.allgather(info)

                stats_per_env = []
                for env_idx in range(0, len(episode_statistics), 5):
                    stats_per_env.append(episode_statistics[env_idx])
                episode_statistics = stats_per_env

                episode_statistics = [episode_statistics[i]['average'] for i in range(len(episode_statistics))]

                mean_statistics = {}
                std_statistics = {}
                all_statistics = {}
                for key in episode_statistics[0]:
                    list_of_stats = list(episode_statistics[idx][key] for idx in range(len(episode_statistics)))
                    mean_statistics[key] = np.average(list_of_stats)
                    std_statistics[key] = np.std(list_of_stats)
                    all_statistics[key] = list_of_stats

                reward_per_env = []
                for env_idx in range(0, len(episode_reward), 5):
                    reward_per_env.append(sum(episode_reward[env_idx:env_idx+5]))

                reward_mean = np.average(reward_per_env)
                reward_std = np.std(reward_per_env)
                running_reward_mean += reward_mean
                running_reward_std += reward_std

                episode_length = np.average(episode_length)

                if policy_record is not None:
                    policy_record.add_result(reward_mean, episode_length)
                    policy_record.save()

                eplen = 0
                epret = 0
                num_dones += 1

                if num_dones == 1 or num_dones % 25 == 0:
                    if policy_record is not None:
                        #with open(os.path.join(policy_record.data_dir, 'accumulated_reward.json'), 'w+') as f:
                        #    json.dump({'mean': running_reward_mean/(num_dones+1), 'std': running_reward_std/(num_dones+1)}, f)
                        with open(os.path.join(policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                            json.dump(mean_statistics, f, indent=True)
                        with open(os.path.join(policy_record.data_dir, 'std_statistics.json'), 'w+') as f:
                            json.dump(std_statistics, f, indent=True)
                        #with open(os.path.join(policy_record.data_dir, 'all_statistics.json'), 'w+') as f:
                        #    json.dump(all_statistics, f, indent=True)


    def learn(self, policy_record, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
              target_kl=0.01, logger_kwargs=dict(), save_freq=-1, tsboard_freq=-1, use_neg_weight=True,
              neg_weight_constant=-1, curriculum_start=-1, curriculum_stop=-1, use_value_norm=False, use_huber_loss=False,
              transfer=False, use_rnn=False, pi_scheduler='constant', vf_scheduler='constant', **kargs):
        """
        Proximal Policy Optimization (by clipping),

        with early stopping based on approximate KL

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with a
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
                module. The ``step`` method should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``v``        (batch,)          | Numpy array of value estimates
                                               | for the provided observations.
                ``logp_a``   (batch,)          | Numpy array of log probs for the
                                               | actions in ``a``.
                ===========  ================  ======================================

                The ``act`` method behaves the same as ``step`` but only returns ``a``.

                The ``pi`` module's forward call should accept a batch of
                observations and optionally a batch of actions, and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       N/A               | Torch Distribution object, containing
                                               | a batch of distributions describing
                                               | the policy for the provided observations.
                ``logp_a``   (batch,)          | Optional (only returned if batch of
                                               | actions is given). Tensor containing
                                               | the log probability, according to
                                               | the policy, of the provided actions.
                                               | If actions not given, will contain
                                               | ``None``.
                ===========  ================  ======================================

                The ``v`` module's forward call should accept a batch of observations
                and return:

                ===========  ================  ======================================
                djymbol       Shape             Description
                ===========  ================  ======================================
                ``v``        (batch,)          | Tensor containing the value estimates
                                               | for the provided observations. (Critical:
                                               | make sure to flatten this!)
                ===========  ================  ======================================


            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while
                still profiting (improving the objective function)? The new policy
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`.

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used
                for early stopping. (Usually small, 0.01 or 0.05.)

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        if tsboard_freq == -1:
            if steps_to_run > 100000:
                tsboard_freq = steps_to_run // 10000
            else:
                tsboard_freq = steps_to_run // 100

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        root = 0
        comm = self.comm
        env = self.env
        setup_pytorch_for_mpi(comm)
        mpi_print(self.env.observation_space, self.env.action_space)

        # Set up logger and save configuration
        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())

        # Random seed
        if seed == -1:
            MAX_INT = 2147483647
            seed = np.random.randint(MAX_INT)
        else:
            if isinstance(seed, list) and len(seed) == 1:
                seed = seed[0]
            seed = seed + proc_id(comm)
        torch.manual_seed(seed)
        np.random.seed(seed)
        mpi_print(proc_id(comm), seed)

        # Instantiate environment
        # augment_obs = gym.spaces.Box(0,1,[env.observation_space.shape[0] + 32])
        # obs_dim = augment_obs.shape
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        o, ep_ret, ep_len = env.reset(), 0, 0
        ep_ret_scheduler = 0
        if use_rnn:
            state_history = [o, o, o, o, o]
            ac_kwargs['use_rnn'] = True

        # Create actor-critic module
        mpi_print(env.observation_space, env.action_space)
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        if transfer:
            state_dict = torch.load(self.kargs['model_path'])
            ac.load_state_dict(state_dict)
        elif self.kargs['model_path']:
            state_dict = torch.load(self.kargs['model_path'])

            temp_state_dict = {}
            for key in state_dict:
                if 'cnn_net' in key:
                    temp_state_dict[key] = state_dict[key]

            ac.load_state_dict(temp_state_dict, strict=False)

            for name, param in ac.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False

        # Sync params across processes
        ac = ac.to(device)
        sync_params(comm, ac, root=root)
        # ac.pi = ac.pi.to(device)
        # ac.v = ac.v.to(device)
        n_process = comm.Get_size()

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        # local_steps_per_epoch = int(steps_per_epoch / num_procs(comm))
        buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, use_rnn=use_rnn)

        if use_value_norm:
            value_normalizer = valuenorm.ValueNorm(1)

        # Set up function for computing PPO policy loss
        def compute_loss_pi(data):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

            # Policy loss
            pi, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            del data, obs, act, adv, logp_old
            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            if not use_value_norm:
                if use_huber_loss:
                    error = ret - ac.v(obs)
                    return huber_loss(error, 10.0).mean()
                else:
                    return ((ac.v(obs) - ret) ** 2).mean()
            else:
                values = ac.v(obs)
                value_normalizer.update(ret)
                error = value_normalizer.normalize(ret) - values
                if use_huber_loss:
                    value_loss = huber_loss(error, 10.0)
                else:
                    value_loss = (error ** 2).mean()
                return value_loss.mean()


        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
        if pi_scheduler == 'smart':
            scheduler_policy = ReduceLROnPlateau(pi_optimizer, mode='max', patience=100, factor=0.05, min_lr=1e-6)
        if vf_scheduler == 'smart':
            scheduler_value = ReduceLROnPlateau(vf_optimizer, mode='max', patience=100, factor=0.05, min_lr=1e-5)

        # Set up model saving
        # logger.setup_pytorch_saver(ac)

        def linear_lr(optimizer, start, epoch, stop):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            init_lr = optimizer.param_groups[0]['lr']
            if init_lr <= stop:
                return
            lr = init_lr - init_lr * (start * 1.0/epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def sqrt_lr(optimizer, start, epoch, stop):
            init_lr = optimizer.param_groups[0]['lr']
            if init_lr <= stop:
                return
            lr = init_lr - init_lr * (start * 1.0 / sqrt(epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_p_index, loss_v_index = 0, 0

        def update():
            #mpi_print('process', comm.Get_rank(), 'run update')
            nonlocal loss_p_index, loss_v_index
            data = buf.get(comm)

            pi_l_old, pi_info_old = compute_loss_pi(data)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(data).item()

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                kl = mpi_avg(comm, pi_info['kl'])
                if kl > 1.5 * target_kl:
                    # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    break
                loss_pi.backward()
                if comm.Get_rank() == root:
                    loss_p_index += 1
                    # writer.add_scalar('loss/Policy loss', loss_pi, loss_p_index)
                mpi_avg_grads(comm, ac.pi)  # average grads across MPI processes
                pi_optimizer.step()
            # Value function learning
            for i in range(train_v_iters):
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data)
                loss_v.backward()
                if comm.Get_rank() == root:
                    loss_v_index += 1
                    # writer.add_scalar('loss/value loss', loss_v, loss_v_index)
                mpi_avg_grads(comm, ac.v)  # average grads across MPI processes
                vf_optimizer.step()

            # Log changes from update
            # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

            del data, pi_info, pi_info_old

        # Prepare for interaction with environment
        start_time = time.time()

        # Main loop: collect experience in env and update/log each epoch

        if comm.Get_rank() == root:
            from pathlib import Path
            Path(os.path.join('./models', str(kargs['model_id']))).mkdir(parents=True, exist_ok=True)
            # writer = SummaryWriter('tsboard/' + date_time_str)
        episode_count = 0

        last_ep_ret = 0
        # ac.load_state_dict(torch.load('models/20210702-150603/5800'))
        # ac.eval()
        ep_ret_mod = 0
        total_step = 0
        last_ep_ret_mod = 0

        step = 0

        last_barrier_encode = None
        last_pos_ret, pos_ret, last_neg_ret, neg_ret = 0, 0, 0, 0

        episode_lengths = []
        episode_returns = []

        if curriculum_start >= 0.0:
            env.friendly_fire_weight = curriculum_start
            period = (curriculum_stop - curriculum_start) // 0.05
            period = int(steps_to_run // period)
            assert curriculum_stop >= curriculum_start

        while step < steps_to_run:

            if curriculum_start >= 0.0 and (step+1) % period == 0:
                env.friendly_fire_weight += 0.05
                env.friendly_fire_weight = min(env.friendly_fire_weight, curriculum_stop)

            if comm.Get_rank() == 0 and step % 25000 == 0:
                #model_path = os.path.join('./models', str(kargs['model_id']), str(step * n_process)+'.pth')
                model_path = os.path.join('./models', str(kargs['model_id']), str(step)+'.pth')
                #mpi_print('save ', model_path)
                torch.save(ac.state_dict(), model_path)

            step += 1
            if use_rnn:
                o = [torch.as_tensor(o, dtype=torch.float32).to(device) for o in state_history]
                o = torch.cat(o, dim=0)
                a, v, logp = ac.step(o)
            else:
                o = torch.as_tensor(o, dtype=torch.float32).to(device)
                a, v, logp = ac.step(o)
            next_o, r, terminal, _ = env.step(a)

            ep_ret += r
            ep_len += 1
            total_step += 1
            ep_ret_scheduler += r

            if r < 0 and use_neg_weight:
                neg_ret += r
            else:
                pos_ret += r
            ep_ret_mod += r
            # save and log
            buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            if use_rnn:
                state_history = state_history[1:] + [next_o]
            else:
                o = next_o

            epoch_ended = step > 0 and step % steps_per_epoch == 0

            if terminal or epoch_ended:
                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)

                if epoch_ended and not (terminal):
                    pass
                    #print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    if use_rnn:
                        o = [torch.as_tensor(o, dtype=torch.float32).to(device) for o in state_history]
                        o = torch.cat(o, dim=0)
                        _, v, _ = ac.step(o)
                    else:
                        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    last_ep_ret = ep_ret
                    last_ep_ret_mod = ep_ret_mod
                    last_neg_ret = neg_ret
                    last_pos_ret = pos_ret

                    o, ep_ret, ep_len = env.reset(), 0, 0
                    o = torch.as_tensor(o, dtype=torch.float32).to(device)
                    if use_rnn:
                        state_history = [o, o, o, o, o]
                    ep_ret_mod, neg_ret, pos_ret = 0, 0, 0

                ep_ret, ep_len = 0, 0

            if epoch_ended:
                update()
                if pi_scheduler == 'lin':
                    linear_lr(pi_optimizer, start=0.5, epoch=step, stop=1e-6)
                elif pi_scheduler == 'sqrt':
                    sqrt_lr(pi_optimizer, start=0.5, epoch=step, stop=1e-6)
                elif pi_scheduler == 'smart':
                    scheduler_policy.step(ep_ret_scheduler)
                if vf_scheduler == 'lin':
                    linear_lr(vf_optimizer, start=0.5, epoch=step, stop=1e-5)
                elif vf_scheduler == 'sqrt':
                    sqrt_lr(vf_optimizer, start=0.5, epoch=step, stop=1e-5)
                elif vf_scheduler == 'smart':
                    scheduler_value.step(ep_ret_scheduler)
                ep_ret_scheduler = 0
                # if comm.Get_rank() == 0:
                #    barrier_encode = ac.pi.cnn(ac.pi.barrier_img)
                #    if last_barrier_encode == None:
                #        last_barrier_encode = barrier_encode
                #    else:
                #        encode_loss = torch.nn.functional.mse_loss(barrier_encode, last_barrier_encode)
                #        writer.add_scalar('record/encode barrier', encode_loss, step * n_process)
                #        writer.add_scalar('record/encode sum abs', torch.sum(torch.abs(encode_loss)), step * n_process)
                #        last_barrier_encode = barrier_encode

            if step % tsboard_freq == 0:
                lrlocal = (episode_lengths, episode_returns)
                listoflrpairs = comm.allgather(lrlocal)
                lens, rews = map(flatten_lists, zip(*listoflrpairs))

                if policy_record is not None:
                    for idx in range(len(lens)):
                        policy_record.add_result(rews[idx], lens[idx])

                    policy_record.save()

                episode_lengths = []
                episode_returns = []
                del lrlocal, listoflrpairs, lens, rews
            '''
            if step % tsboard_freq == 0:
                ep_ret_gather = comm.gather(last_ep_ret, root=0)
                ep_ret_mod_gather = comm.gather(last_ep_ret_mod , root=0)
                pos_ret_gather = comm.gather(last_pos_ret, root=0)
                neg_ret_gather = comm.gather(last_neg_ret, root=0)
                if comm.Get_rank() == root:
                    pol_total_step = step * n_process
                    writer.add_scalar('return/no weight', np.mean(ep_ret_gather), pol_total_step)
                    writer.add_scalar('return/curriculum', np.mean(ep_ret_mod_gather), pol_total_step)
                    writer.add_scalar('return/neg', np.mean(neg_ret_gather), pol_total_step)
                    writer.add_scalar('return/pos', np.mean(pos_ret_gather), pol_total_step)
                    writer.add_scalar('record/neg weight', neg_weight, pol_total_step)
            '''


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
