# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import pdb
import pickle, os
from shutil import copy
import matplotlib.pyplot as plt

from core.plot_utils import *


def get_dir_for_policy(policy_id, log_comms_dir):
    #return log_comms_dir + "policy_"+str(policy_id)+"/"
    return log_comms_dir + str(policy_id) + "/"


class RecordChannel():
    def __init__(self, data_dir, color='#eb0033', ylabel='Episodic Damage',
                 windows=[20,50,100], alphas=[0.1,0.3,1.0], intrinsic_reward=False,
                 std=False):

        self.data_dir = data_dir
        self.ep_results = []
        self.ep_red_blue_damages = []
        self.ep_red_red_damages = []
        self.ep_blue_red_damages = []
        if intrinsic_reward:
            self.ep_intrinsic = []
        self.ep_lengths = []
        self.ep_cumlens = []
        self.color = color
        self.ylabel = ylabel
        self.windows = windows
        self.alphas = alphas
        self.intrinsic_reward = intrinsic_reward
        self.std = std
        if std:
            self.ep_stds = [[], [], []]

    def save(self):
        if self.intrinsic_reward:
            data = [self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages,
                    self.ep_intrinsic, self.ep_lengths, self.ep_cumlens]
        elif self.std:
            data = [self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages,
                    self.ep_stds, self.ep_lengths, self.ep_cumlens]
        else:
            data = [self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages,
                    self.ep_lengths, self.ep_cumlens]
        pickle.dump(data, open(self.data_dir+'policy_record.p', 'wb'))

        # save a plot also
        plot_policy_records_damage([self], self.windows, self.alphas,
                                   self.data_dir + "plot_damage.png", colors=[self.color],
                                   episodic=True)
        plot_policy_records([self], self.windows, self.alphas,
                            self.data_dir + "plot_reward.png", colors=[self.color],
                            episodic=False)

        if self.intrinsic_reward:
            plot_policy_records([self], self.windows, self.alphas,
                            self.data_dir + "plot_intrinsic_reward.png", colors=[self.color],
                            episodic=False, intrinsic=True)

        if self.std:
            plot_policy_records_std([self], self.windows, self.alphas,
                                    self.data_dir + "plot_std.png", colors=[self.color],
                                    episodic=False)

    def load(self):
        path = self.data_dir + "policy_record.p"
        if os.path.exists(path):
            data = pickle.load(open(path, "rb"))
            if self.intrinsic_reward:
                self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages, \
                    self.ep_intrinsic, self.ep_lengths, self.ep_cumlens = data
            elif self.std:
                self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages, \
                    self.ep_stds, self.ep_lengths, self.ep_cumlens = data
            else:
                self.ep_results, self.ep_red_red_damages, self.ep_red_blue_damages, self.ep_blue_red_damages, \
                    self.ep_lengths, self.ep_cumlens = data

    def get_copy(self):
        rc = RecordChannel(self.data_dir,
                           color=self.color,
                           ylabel=self.ylabel,
                           windows=self.windows,
                           alphas=self.alphas)

        rc.ep_results = self.ep_results[:]
        rc.ep_red_red_damages = self.ep_red_red_damages[:]
        rc.ep_blue_red_damages = self.ep_blue_red_damages[:]
        rc.ep_red_blue_damages = self.ep_red_blue_damages[:]
        rc.ep_lengths = self.ep_lengths[:]
        rc.ep_cumlens = self.ep_cumlens[:]
        if self.intrinsic_reward:
            rc.ep_intrinsic = self.ep_intrinsic[:]
        if self.std:
            rc.ep_stds = self.ep_stds[:]
        return rc


class PolicyRecord():

    def __init__(self, policy_folder_name, log_comms_dir, plot_color="#eb0033", intrinsic_reward=False, std=False):

        self.plot_color = plot_color

        self.log_comms_dir = log_comms_dir
        self.data_dir = get_dir_for_policy(policy_folder_name, log_comms_dir)
        self.channels = {'main': RecordChannel(self.data_dir, color=plot_color, intrinsic_reward=intrinsic_reward,
                                               std=std)}

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        else:
            self.load()

    def add_result(self, total_reward, ep_len, intrinsic_reward=None, channel='main'):

        if channel not in self.channels:
            self.add_channel(channel)

        ch = self.channels[channel]

        ch.ep_results.append(total_reward)
        if intrinsic_reward:
            ch.ep_intrinsic.append(intrinsic_reward)
        ch.ep_lengths.append(ep_len)
        if len(ch.ep_cumlens) == 0:
            ch.ep_cumlens.append(ep_len)
        else:
            ch.ep_cumlens.append(ch.ep_cumlens[-1] + ep_len)

    def add_damage_result(self, red_blue_damage, red_red_damage, blue_red_damage, channel='main'):

        if channel not in self.channels:
            self.add_channel(channel)

        ch = self.channels[channel]

        ch.ep_red_blue_damages.append(red_blue_damage)
        ch.ep_blue_red_damages.append(blue_red_damage)
        ch.ep_red_red_damages.append(red_red_damage)

    '''
    def add_result(self, total_reward, red_blue_damage, red_red_damage, blue_red_damage, ep_len, channel="main",
                   intrinsic_reward=None, std=None):

        if channel not in self.channels:
            self.add_channel(channel)

        ch = self.channels[channel]

        ch.ep_results.append(total_reward)
        ch.ep_red_blue_damages.append(red_blue_damage)
        ch.ep_blue_red_damages.append(blue_red_damage)
        ch.ep_red_red_damages.append(red_red_damage)
        ch.ep_lengths.append(ep_len)
        if len(ch.ep_cumlens) == 0:
            ch.ep_cumlens.append(ep_len)
        else:
            ch.ep_cumlens.append(ch.ep_cumlens[-1]+ep_len)
        if intrinsic_reward is not None:
            ch.ep_intrinsic.append(intrinsic_reward)
        if std is not None:
            ch.ep_stds[0].append(std[0])
            ch.ep_stds[1].append(std[1])
            ch.ep_stds[2].append(std[2])
    '''


    def add_channel(self, channel_name, **kwargs):

        if channel_name not in self.channels:
            ch = RecordChannel(self.data_dir, name=channel_name, **kwargs)
            ch.load()
            self.channels[channel_name] = ch


    def save(self):
        for ch in self.channels:
            self.channels[ch].save()


    def load(self):
        for ch in self.channels:
            self.channels[ch].load()


    def fork(self, new_id):
        new_pr = PolicyRecord(new_id, self.log_comms_dir)

        for ch in self.channels:
            new_ch = self.channels[ch].get_copy()
            new_pr.channels[ch] = new_ch
        
        # copy files over from existing log dir
        for f in os.listdir(self.data_dir):

            # HACK: check for f not having any extension
            if "." not in f:
                continue

            copy(self.data_dir+f, new_pr.data_dir)

        return new_pr
