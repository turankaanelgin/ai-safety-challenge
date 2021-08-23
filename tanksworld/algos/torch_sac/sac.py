from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from . import core
from algos.torch_ppo.utils.mpi_pytorch import setup_pytorch_for_mpi

from stable_baselines.trpo_mpi.utils import flatten_lists


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class SACPolicy():
    def __init__(self, env, policy_comm, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.comm = policy_comm
        self.eval_mode = eval_mode

    def run(self, num_steps, data_dir, policy_record=None):

        if self.eval_mode:
            self.evaluate(policy_record, num_steps, **self.kargs)
        else:
            self.learn(policy_record, **self.kargs)

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

        while steps < local_steps:
            # observation = np.array(observation).reshape(-1, 72)
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

                temp_statistics = {}
                for key in episode_statistics[0]:
                    temp_statistics[key] = np.average(list(episode_statistics[idx][key] \
                                                           for idx in range(len(episode_statistics))))
                episode_statistics = temp_statistics

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

                if num_dones % 50 == 0:
                    if policy_record is not None:
                        with open(os.path.join(policy_record.data_dir, 'accumulated_reward.json'), 'w+') as f:
                            json.dump({'mean': running_reward_mean / (num_dones + 1),
                                       'std': running_reward_std / (num_dones + 1)}, f)
                        with open(os.path.join(policy_record.data_dir, 'game_statistics.json'), 'w+') as f:
                            json.dump(episode_statistics, f)

                num_dones += 1


    def learn(self, policy_record, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly: gradients
                                               | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        comm = self.comm
        setup_pytorch_for_mpi(comm)

        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=steps_per_epoch)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

        # Set up function for computing SAC Q-losses
        def compute_loss_q(data):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            q1 = ac.q1(o,a)
            q2 = ac.q2(o,a)

            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, logp_a2 = ac.pi(o2)

                # Target Q-values
                q1_pi_targ = ac_targ.q1(o2, a2)
                q2_pi_targ = ac_targ.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            q_info = dict(Q1Vals=q1.detach().numpy(),
                          Q2Vals=q2.detach().numpy())

            return loss_q, q_info

        # Set up function for computing SAC pi loss
        def compute_loss_pi(data):
            o = data['obs']
            pi, logp_pi = ac.pi(o)
            q1_pi = ac.q1(o, pi)
            q2_pi = ac.q2(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)

            # Entropy-regularized policy loss
            loss_pi = (alpha * logp_pi - q_pi).mean()

            # Useful info for logging
            pi_info = dict(LogPi=logp_pi.detach().numpy())

            return loss_pi, pi_info

        # Set up optimizers for policy and q-function
        pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
        q_optimizer = Adam(q_params, lr=lr)

        # Set up model saving

        def update(data):
            # First run one gradient descent step for Q1 and Q2
            q_optimizer.zero_grad()
            loss_q, q_info = compute_loss_q(data)
            loss_q.backward()
            q_optimizer.step()

            # Record things

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        def get_action(o, deterministic=False):
            return ac.act(torch.as_tensor(o, dtype=torch.float32),
                          deterministic)

        def test_agent():
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                while not(d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time
                    o, r, d, _ = test_env.step(get_action(o, True))
                    ep_ret += r
                    ep_len += 1

        # Prepare for interaction with environment
        total_steps = steps_per_epoch * epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        episode_lengths = []
        episode_returns = []

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > start_steps:
                a = get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)

            if (t+1) % steps_per_epoch == 0:
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
            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs):
                    logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                test_agent()

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
            '''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
