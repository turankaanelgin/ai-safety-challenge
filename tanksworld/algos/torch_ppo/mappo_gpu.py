import pdb
import pprint
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CyclicLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from . import core
import os
import json
from matplotlib import pyplot as plt

from algos.torch_ppo.mappo_utils import valuenorm
from algos.torch_ppo.mappo_utils.util import huber_loss


device = torch.device('cuda')


class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.005    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                               (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                           (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


class PPOBufferCUDAUpdated:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, n_envs=1, use_rnn=False, n_states=3):
        if use_rnn:
            self.obs_buf = torch.zeros(core.combined_shape_v4(size, n_envs, 5, n_states, obs_dim)).to(device)
        else:
            self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, obs_dim)).to(device)
        self.act_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.rew_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.ret_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.val_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.logp_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.entropy_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        self.masks = torch.ones((size+1, n_envs, 5)).to(device)
        self.active_masks = torch.ones_like(self.masks)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.n_envs = n_envs
        self.use_rnn = use_rnn

    def store(self, obs, act, rew, val, logp, entropy, terminal, active_mask):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        if self.use_rnn:
            self.obs_buf[self.ptr] = obs
        else:
            self.obs_buf[self.ptr] = obs.squeeze(2)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.entropy_buf[self.ptr] = entropy
        self.masks[self.ptr+1] = 1-torch.cuda.LongTensor(terminal)
        self.active_masks[self.ptr+1] = active_mask
        self.ptr += 1

    def finish_path(self, last_val):

        path_slice = slice(self.path_start_idx, self.ptr)

        last_val = last_val.unsqueeze(0)
        vals = torch.cat((self.val_buf[path_slice], last_val), dim=0)
        rews = self.rew_buf[path_slice]
        gae = 0
        for step in reversed(range(rews.shape[0])):
            delta = rews[step] + self.gamma * vals[step+1] * self.masks[step+1] - vals[step]
            gae = delta + self.gamma * self.lam * self.masks[step+1] * gae
            self.ret_buf[step] = gae + vals[step]

        deltas = rews + self.gamma * vals[1:] - vals[:-1]
        discount_delta = core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)
        self.adv_buf[path_slice] = torch.as_tensor(discount_delta.copy(), dtype=torch.float32).to(device)
        #self.adv_buf[self.active_masks[:-1] == 0.0] = np.nan

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_buf = self.adv_buf.flatten(start_dim=1).cpu().numpy()
        adv_mean = np.nanmean(adv_buf, axis=0)
        adv_std = np.nanstd(adv_buf, axis=0)
        adv_buf = (adv_buf - adv_mean) / (adv_std + 1e-5)
        self.adv_buf = torch.as_tensor(adv_buf.reshape(adv_buf.shape[0], self.n_envs, 5)).to(device)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, entropy=self.entropy_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}


class PPOBufferCUDA:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, n_envs=1, use_rnn=False, n_states=3):
        if use_rnn:
            self.obs_buf = torch.zeros(core.combined_shape_v4(size, n_envs, 5, n_states, obs_dim)).to(device)
        else:
            self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, obs_dim)).to(device)
        self.act_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.rew_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.ret_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.val_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.logp_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.entropy_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.n_envs = n_envs
        self.use_rnn = use_rnn

    def store(self, obs, act, rew, val, logp, entropy):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        if self.use_rnn:
            self.obs_buf[self.ptr] = obs
        else:
            self.obs_buf[self.ptr] = obs.squeeze(2)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.entropy_buf[self.ptr] = entropy
        self.ptr += 1

    def finish_path(self, last_val=[0, 0, 0, 0, 0]):
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

        last_val = last_val.unsqueeze(0)
        rews = torch.cat((self.rew_buf[path_slice], last_val), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_val), dim=0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        discount_delta = core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)
        self.adv_buf[path_slice] = torch.as_tensor(discount_delta.copy(), dtype=torch.float32).to(device)

        # the next line computes rewards-to-go, to be targets for the value function
        discount_rews = core.discount_cumsum(rews.cpu().numpy(), self.gamma)[:-1]
        self.ret_buf[path_slice] = torch.as_tensor(discount_rews.copy(), dtype=torch.float32).to(device)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_buf = self.adv_buf.flatten(start_dim=1)
        adv_std, adv_mean = torch.std_mean(adv_buf, dim=0)
        adv_buf = (adv_buf - adv_mean) / adv_std
        self.adv_buf = adv_buf.reshape(adv_buf.shape[0], self.n_envs, 5)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, entropy=self.entropy_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}


class PPOPolicy():
    def __init__(self, env, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.eval_mode = eval_mode

    def run(self, policy_record, num_steps):

        if self.eval_mode:
            self.evaluate(policy_record, num_steps, **self.kargs)
        else:
            self.kargs.update({'steps_to_run': num_steps})
            self.learn(policy_record, **self.kargs)

    def evaluate(self, policy_record, total_timesteps, ac=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),
                 **kargs):

        # local_steps = int(total_timesteps / self.comm.Get_size())
        steps = 0
        observation = self.env.reset()

        if not ac:
            ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
            ac.load_state_dict(torch.load(self.kargs['model_path'])['model_state_dict'], strict=True)
        ac.eval()

        eplen = 0
        epret = 0
        running_reward_mean = 0.0
        running_reward_std = 0.0
        num_dones = 0

        pp = pprint.PrettyPrinter(indent=4)

        #mpi_print('Computing the metrics...')

        while steps < total_timesteps:
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
                    reward_per_env.append(sum(episode_reward[env_idx:env_idx + 5]))

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

                if num_dones == 1 or num_dones == total_timesteps or num_dones % 25 == 0:
                    if policy_record is not None:
                        with open(os.path.join(policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                            json.dump(mean_statistics, f, indent=True)
                        with open(os.path.join(policy_record.data_dir, 'std_statistics.json'), 'w+') as f:
                            json.dump(std_statistics, f, indent=True)


    def save_metrics(self, episode_statistics, policy_record, step):

        length = len(episode_statistics[0]['all'])
        if length < 3: return

        episode_statistics = [episode_statistics[env_idx]['all'][-min(100, length):] \
                              for env_idx in range(len(episode_statistics))]

        for env_idx in range(len(episode_statistics)):
            episode_statistics[env_idx] = {key: np.average([episode_statistics[env_idx][i][key] \
                                                           for i in range(len(episode_statistics[env_idx]))]) \
                                                           for key in episode_statistics[env_idx][0]}

        mean_statistics = {}
        std_statistics = {}
        all_statistics = {}
        for key in episode_statistics[0]:
            list_of_stats = [episode_statistics[idx][key] for idx in range(len(episode_statistics))]
            mean_statistics[key] = np.average(list_of_stats)
            std_statistics[key] = np.std(list_of_stats)
            all_statistics[key] = list_of_stats

        if policy_record is not None:
            with open(os.path.join(policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                json.dump(mean_statistics, f, indent=True)
            with open(os.path.join(policy_record.data_dir, 'std_statistics.json'), 'w+') as f:
                json.dump(std_statistics, f, indent=True)


    def learn(self, policy_record, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, ent_coef=0.0, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, tsboard_freq=-1, curriculum_start=-1, curriculum_stop=-1, use_value_norm=False,
              use_huber_loss=False, use_rnn=False, use_popart=False, pi_scheduler='cons', vf_scheduler='cons',
              freeze_rep=True, **kargs):
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
                tsboard_freq = steps_to_run // 1000

        env = self.env
        tb = SummaryWriter()

        # Random seed
        if seed == -1:
            MAX_INT = 2147483647
            seed = np.random.randint(MAX_INT)
        else:
            if isinstance(seed, list) and len(seed) == 1:
                seed = seed[0]

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        o, ep_ret, ep_len = env.reset(), 0, 0
        ep_ret_scheduler, ep_len_scheduler = 0, 0
        num_states = kargs['num_states'] if 'num_states' in kargs else None
        if use_rnn:
            assert num_states is not None
            state_history = [o] * num_states
            ac_kwargs['use_rnn'] = True
        if use_popart:
            ac_kwargs['use_popart'] = True

        #ac_kwargs['use_sde'] = True # TODO remove
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
        start_step = 0

        scheduler_policy = None
        if pi_scheduler == 'smart':
            scheduler_policy = ReduceLROnPlateau(pi_optimizer, mode='max', factor=0.5, patience=1000)
        elif pi_scheduler == 'linear':
            scheduler_policy = LinearLR(pi_optimizer, start_factor=0.5, end_factor=1.0, total_iters=5000)
        elif pi_scheduler == 'cyclic':
            scheduler_policy = CyclicLR(pi_optimizer, base_lr=pi_lr, max_lr=3e-4, mode='triangular2', cycle_momentum=False)

        scheduler_value = None
        if vf_scheduler == 'smart':
            scheduler_value = ReduceLROnPlateau(vf_optimizer, mode='max', factor=0.5, patience=1000)
        elif vf_scheduler == 'linear':
            scheduler_value = LinearLR(vf_optimizer, start_factor=0.5, end_factor=1.0, total_iters=5000)
        elif vf_scheduler == 'cyclic':
            scheduler_value = CyclicLR(vf_optimizer, base_lr=vf_lr, max_lr=1e-3, mode='triangular2', cycle_momentum=False)

        assert pi_scheduler != 'cons' and scheduler_policy or not scheduler_policy
        assert vf_scheduler != 'cons' and scheduler_value or not scheduler_value

        # Load from previous checkpoint
        if self.kargs['model_path']:
            ckpt = torch.load(self.kargs['model_path'])
            ac.load_state_dict(ckpt['model_state_dict'], strict=True)
            pi_optimizer.load_state_dict(ckpt['pi_optimizer_state_dict'])
            vf_optimizer.load_state_dict(ckpt['vf_optimizer_state_dict'])
            if pi_scheduler != 'cons':
                scheduler_policy.load_state_dict(ckpt['pi_scheduler_state_dict'])
            if vf_scheduler != 'cons':
                scheduler_value.load_state_dict(ckpt['vf_scheduler_state_dict'])
            start_step = ckpt['step']
            start_step -= start_step % steps_per_epoch

            if freeze_rep:
                for name, param in ac.named_parameters():
                    if 'cnn_net' in name:
                        param.requires_grad = False

        # Only load the representation part
        elif self.kargs['cnn_model_path'] and freeze_rep:
            state_dict = torch.load(self.kargs['cnn_model_path'])

            temp_state_dict = {}
            for key in state_dict:
                if 'cnn_net' in key:
                    temp_state_dict[key] = state_dict[key]

            ac.load_state_dict(temp_state_dict, strict=False)

            for name, param in ac.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False

        buf = PPOBufferCUDA(obs_dim, act_dim, steps_per_epoch, gamma, lam, n_envs=kargs['n_envs'],
                            use_rnn=use_rnn, n_states=num_states)

        if use_popart:
            value_normalizer = ac.v.v_net
        else:
            value_normalizer = None

        # Set up function for computing PPO policy loss
        def compute_loss_pi(data):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            obs, act, adv, logp_old = obs.to(device), act.to(device), adv.to(device), logp_old.to(device)
            obs = torch.flatten(obs, end_dim=2)
            if use_rnn:
                act = torch.flatten(act, end_dim=1)
            else:
                act = torch.flatten(act, end_dim=2)

            # Policy loss
            pi, logp = ac.pi(obs, act)
            logp = logp.reshape(steps_per_epoch, kargs['n_envs'], 5)
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
            obs, ret = obs.to(device), ret.to(device)
            obs = torch.flatten(obs, end_dim=2)
            ret = torch.flatten(ret)

            if not use_value_norm:
                if use_huber_loss:
                    error = ret - ac.v(obs)
                    return huber_loss(error, 10.0).mean()
                else:
                    if use_rnn:
                        return ((ac.v(obs).flatten() - ret) ** 2).mean()
                    else:
                        return ((ac.v(obs).squeeze(0) - ret) ** 2).mean()
            elif use_popart:
                values = ac.v(obs)
                value_normalizer.update(ret)
                error = value_normalizer.normalize(ret) - values
                return (error ** 2).mean()
            else:
                values = ac.v(obs)
                value_normalizer.update(ret)
                error = value_normalizer.normalize(ret) - values
                if use_huber_loss:
                    value_loss = huber_loss(error, 10.0)
                else:
                    value_loss = (error ** 2).mean()
                del data
                return value_loss.mean()

        # Set up function for computing entropy loss
        def compute_loss_entropy(data):
            entropy_loss = -torch.mean(data['entropy'])
            return entropy_loss

        loss_p_index, loss_v_index = 0, 0

        def update():
            nonlocal loss_p_index, loss_v_index
            nonlocal step
            nonlocal writer
            data = buf.get()

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                if ent_coef > 0.0:
                    loss_entropy = compute_loss_entropy(data)
                kl = pi_info['kl']
                if kl > 1.5 * target_kl:
                    # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    break
                if ent_coef > 0.0:
                    loss = loss_pi + ent_coef * loss_entropy
                else:
                    loss = loss_pi
                loss.backward()
                loss_p_index += 1
                writer.add_scalar('loss/Policy loss', loss_pi, loss_p_index)
                if ent_coef > 0.0:
                    writer.add_scalar('loss/Entropy loss', loss_entropy, loss_p_index)
                pi_optimizer.step()
            # Value function learning
            for i in range(train_v_iters):
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data)
                loss_v.backward()
                loss_v_index += 1
                writer.add_scalar('loss/value loss', loss_v, loss_v_index)
                vf_optimizer.step()
            #writer.add_scalar('std_dev/translation', torch.exp(ac.pi.log_std)[0].item(), step)
            #writer.add_scalar('std_dev/orientation', torch.exp(ac.pi.log_std)[1].item(), step)
            #writer.add_scalar('std_dev/shoot', torch.exp(ac.pi.log_std)[2].item(), step)
            del data, pi_info

        # Prepare for interaction with environment
        # Main loop: collect experience in env and update/log each epoch

        if not os.path.exists(os.path.join(kargs['save_dir'], str(kargs['model_id']))):
            from pathlib import Path
            Path(os.path.join(kargs['save_dir'], str(kargs['model_id']))).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(os.path.join(policy_record.data_dir, 'tsboard'))

        total_step = start_step

        step = start_step

        episode_lengths = []
        episode_returns = []
        policy_learning_rates = [pi_lr]
        value_learning_rates = [vf_lr]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        while step < steps_to_run:

            if (step+1) % 25000 == 0 or step == 0:
                model_path = os.path.join(kargs['save_dir'], str(kargs['model_id']), str(step) + '.pth')
                ckpt_dict = {'step': step,
                             'model_state_dict': ac.state_dict(),
                             'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                             'vf_optimizer_state_dict': vf_optimizer.state_dict()}
                if pi_scheduler != 'cons':
                    ckpt_dict['pi_scheduler_state_dict'] = scheduler_policy.state_dict()
                if vf_scheduler != 'cons':
                    ckpt_dict['vf_scheduler_state_dict'] = scheduler_value.state_dict()
                torch.save(ckpt_dict, model_path)

            step += 1
            if use_rnn:
                o = [torch.as_tensor(o, dtype=torch.float32).to(device) for o in state_history]
                o = torch.cat(o, dim=2)
            else:
                o = torch.as_tensor(o, dtype=torch.float32).to(device)
            a, v, logp, entropy = ac.step(o)
            next_o, r, terminal, info = env.step(a.cpu().numpy())

            ep_ret += sum(r)
            ep_ret_scheduler += sum(r)
            ep_len += 1
            ep_len_scheduler += 1
            total_step += 1

            # save and log
            r = torch.as_tensor(r, dtype=torch.float32).to(device)
            a = torch.as_tensor(a, dtype=torch.float32).to(device)
            v = torch.as_tensor(v, dtype=torch.float32).to(device)
            logp = torch.as_tensor(logp, dtype=torch.float32).to(device)
            if use_popart:
                v = value_normalizer.denormalize(v)
            buf.store(o, a, r, v, logp, entropy)

            # Update obs (critical!)
            o = next_o

            epoch_ended = step > 0 and step % steps_per_epoch == 0

            if np.all(terminal) or epoch_ended:
                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)

                #if epoch_ended and not np.all(terminal):
                #   pass
                #   print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    if use_rnn:
                        o = [torch.as_tensor(o, dtype=torch.float32).to(device) for o in state_history]
                        o = torch.cat(o, dim=2)
                        _, v, _, _ = ac.step(o)
                    else:
                        _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v = torch.zeros((kargs['n_envs'], 5)).to(device)
                buf.finish_path(v)
                if np.all(terminal):
                    o, ep_ret, ep_len = env.reset(), 0, 0
                    o = torch.as_tensor(o, dtype=torch.float32).to(device)
                    if use_rnn:
                        state_history = [o] * num_states

                ep_ret, ep_len = 0, 0

            if epoch_ended:
                update()

                if pi_scheduler == 'smart':
                    scheduler_policy.step(ep_ret_scheduler/ep_len_scheduler)
                elif pi_scheduler != 'cons':
                    scheduler_policy.step()
                writer.add_scalar('learning_rate/Policy LR', pi_optimizer.param_groups[0]['lr'], step)
                policy_learning_rates.append(pi_optimizer.param_groups[0]['lr'])

                if vf_scheduler == 'smart':
                    scheduler_value.step(ep_ret_scheduler/ep_len_scheduler)
                elif vf_scheduler != 'cons':
                    scheduler_value.step()
                writer.add_scalar('learning_rate/Value LR', vf_optimizer.param_groups[0]['lr'], step)
                value_learning_rates.append(vf_optimizer.param_groups[0]['lr'])

                ep_ret_scheduler, ep_len_scheduler = 0, 0

            if step % 50 == 0 or step == 4:
                episode_statistics = info
                self.save_metrics(episode_statistics, policy_record, step=step)

                ax.plot(policy_learning_rates, color='red')
                ax.plot(value_learning_rates, color='blue')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Learning Rate')
                plt.savefig(os.path.join(policy_record.data_dir, 'learning_rate.png'))

            if step % tsboard_freq == 0:
                lens, rews = episode_lengths, episode_returns

                if policy_record is not None:
                    for idx in range(len(lens)):
                        policy_record.add_result(rews[idx], lens[idx])
                    policy_record.save()

                episode_lengths = []
                episode_returns = []

        tb.close()