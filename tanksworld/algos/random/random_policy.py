import random
import json
import numpy as np
import os

from arena5.core.utils import mpi_print
from arena5.wrappers.mpi_logging_wrappers import MPISynchronizedPRUpdater


class RandomPolicy():

    def __init__(self, env, policy_comm):
        self.env = env
        self.comm = policy_comm

    def run(self, num_steps, data_dir, policy_record=None):

        self.env = MPISynchronizedPRUpdater(self.env, self.comm, policy_record)
        self.env.reset()

        local_steps = int(num_steps / self.comm.Get_size())
        eplen = 0
        epret = 0
        running_reward_mean = 0.0
        running_reward_std = 0.0
        num_dones = 0

        for stp in range(local_steps):
            a = self.env.action_space.sample()
            _, reward, done, info = self.env.step(a)

            eplen += 1
            epret += reward

            if done:
                self.env.reset()

                episode_reward = self.comm.allgather(epret)
                episode_length = self.comm.allgather(eplen)
                episode_statistics = self.comm.allgather(info)

                stats_per_env = []
                for env_idx in range(0, len(episode_statistics), 5):
                    stats_per_env.append(episode_statistics[env_idx])
                episode_statistics = stats_per_env

                mean_statistics = {}
                for key in episode_statistics[0]:
                    mean_statistics[key] = np.average(list(episode_statistics[idx][key] \
                                                           for idx in range(len(episode_statistics))))
                std_statistics = {}
                for key in episode_statistics[0]:
                    if key == 'ally_damage_amount_red':
                        stats_list = list(episode_statistics[idx][key] for idx in range(len(episode_statistics)))
                        print('STATS', stats_list)
                        print('STD STATS', np.std(stats_list))
                    std_statistics[key] = np.std(list(episode_statistics[idx][key] \
                                                      for idx in range(len(episode_statistics))))
                all_statistics = {}
                for key in episode_statistics[0]:
                    all_statistics[key] = list(episode_statistics[idx][key] \
                                               for idx in range(len(episode_statistics)))

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
                        with open(os.path.join(policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                            json.dump(mean_statistics, f, indent=True)
                        with open(os.path.join(policy_record.data_dir, 'std_statistics.json'), 'w+') as f:
                            json.dump(std_statistics, f, indent=True)
                        with open(os.path.join(policy_record.data_dir, 'all_statistics.json'), 'w+') as f:
                            json.dump(all_statistics, f, indent=True)

                num_dones += 1
