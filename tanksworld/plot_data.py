import pdb

import matplotlib.pyplot as plt
import os
import json
import numpy as np


#penalties = [0.1, 0.25, 0.7, 0.8, 0.9, 'rand', '0.1-new', '0.25-new', '0.7-new', '0.8-new', '0.9-new']
#penalties = [0.7, 0.8, 0.9, 'rand', 'iter-16-0.8', 'iter-8-0.8', 'ff-0.8-p-0.8', 'ff-1.0-p-0.2', 'anneal-mini-horizon',
#             'ff-0.8-p-0.2-dp', 'ff-0.8-p-0.8-kb']
penalties = ['iter64-125', 'iter16-125', 'iter64-375']
colors = ['black', 'red', 'blue']
#colors = ['black', 'red', 'blue', 'green', 'purple', 'tomato', 'darkkhaki', 'aquamarine', 'lightpink', 'steelblue', 'darkgoldenrod']
#penalties = ['ff-0.8-p-0.8', 'ff-1.0-p-0.2']
red_ally_damage = []
red_enemy_damage = []
blue_ally_damage = []
blue_enemy_damage = []
red_ally_kills = []
red_enemy_kills = []
blue_ally_kills = []
blue_enemy_kills = []
red_ally_kills_mean = []
red_enemy_kills_mean = []
blue_ally_kills_mean = []
blue_enemy_kills_mean = []
red_ally_kills_winning = []
red_enemy_kills_winning = []
blue_ally_kills_losing = []
blue_enemy_kills_losing = []
red_ally_kills_winning_mean = []
red_enemy_kills_winning_mean = []
blue_ally_kills_losing_mean = []
blue_enemy_kills_losing_mean = []
red_ally_damage_winning = []
red_enemy_damage_winning = []
blue_ally_damage_losing = []
blue_enemy_damage_losing = []
red_ally_damage_winning_mean = []
red_enemy_damage_winning_mean = []
blue_ally_damage_losing_mean = []
blue_enemy_damage_losing_mean = []
red_ally_damage_mean = []
red_enemy_damage_mean = []
blue_ally_damage_mean = []
blue_enemy_damage_mean = []
red_ally_damage_std = []
red_enemy_damage_std = []
blue_ally_damage_std = []
blue_enemy_damage_std = []
reward_mean = []
reward_std = []
x_axis = []

'''
i = 0
for folder in ['./logs/friendly-fire-replicated-eval']:
    for pidx in range(1, 4):
        if folder == './logs/friendly-fire-eval' and pidx == 1:
            continue
        if folder == './logs/friendly-fire-eval' and pidx != 2:
            break
        #if folder == './logs/friendly-fire-replicated-eval' and pidx != 1:
        #    break
        policy = 'policy_{}'.format(pidx)
        with open(os.path.join(folder, policy, 'mean_statistics.json'), 'r') as f:
            mean_statistics = json.load(f)
        with open(os.path.join(folder, policy, 'std_statistics.json'), 'r') as f:
            std_statistics = json.load(f)

        red_enemy_damage.append(mean_statistics['red_enemy_damage'])
        red_ally_damage.append(mean_statistics['red_ally_damage'])
        blue_enemy_damage.append(mean_statistics['blue_enemy_damage'])
        red_enemy_damage_std.append(std_statistics['red_enemy_damage'])
        red_ally_damage_std.append(std_statistics['red_ally_damage'])
        blue_enemy_damage_std.append(std_statistics['blue_enemy_damage'])


red_inflicted_damage = red_enemy_damage
red_taken_damage = np.add(np.asarray(blue_enemy_damage), np.asarray(red_ally_damage))
red_inflicted_damage_err = red_enemy_damage_std
red_taken_damage_err = max(blue_enemy_damage_std, red_ally_damage_std)
plt.errorbar(y=red_inflicted_damage, x=red_taken_damage, yerr=red_inflicted_damage_err,
             xerr=red_taken_damage_err, linestyle='None')
for i, (y,x) in enumerate(zip(red_inflicted_damage, red_taken_damage)):
    plt.annotate(penalties[i], (x,y))
plt.plot(np.linspace(0, 600, 100), np.linspace(0, 600, 100), linestyle='dashed')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.xlabel('Damage taken by red')
plt.ylabel('Damage inflicted by red')
plt.title('Damage inflicted by policies')
plt.show()
'''
i = 0
for folder in ['./logs/new-set-of-experiments-eval']:
    for pidx in range(1, 4):
        policy = 'policy_{}'.format(pidx)
        if not os.path.exists(os.path.join(folder, policy)):
            break
        if folder == './logs/friendly-fire-replicated-eval' and pidx != 1:
            break
        if folder == './logs/frozen-cnn-eval' and (pidx == 1 or pidx == 2):
            continue
        try:
            with open(os.path.join(folder, policy, 'all_statistics.json'), 'r') as f:
                all_statistics = json.load(f)
        except:
            pass
        with open(os.path.join(folder, policy, 'mean_statistics.json'), 'r') as f:
            mean_statistics = json.load(f)
        with open(os.path.join(folder, policy, 'std_statistics.json'), 'r') as f:
            std_statistics = json.load(f)
        try:
            red_ally_damage += all_statistics['ally_damage_amount_red']
            red_enemy_damage += all_statistics['enemy_damage_amount_red']
            blue_ally_damage += all_statistics['ally_damage_amount_blue']
            blue_enemy_damage += all_statistics['enemy_damage_amount_blue']
        except:
            pass
        red_ally_damage_mean.append(mean_statistics['ally_damage_amount_red'])
        red_enemy_damage_mean.append(mean_statistics['enemy_damage_amount_red'])
        blue_ally_damage_mean.append(mean_statistics['ally_damage_amount_blue'])
        blue_enemy_damage_mean.append(mean_statistics['enemy_damage_amount_blue'])
        red_ally_damage_std.append(std_statistics['ally_damage_amount_red'])
        red_enemy_damage_std.append(std_statistics['enemy_damage_amount_red'])
        blue_ally_damage_std.append(std_statistics['ally_damage_amount_blue'])
        blue_enemy_damage_std.append(std_statistics['enemy_damage_amount_blue'])
        try:
            red_ally_damage_winning += all_statistics['red_winning_episode_red_ally_damage']
            red_enemy_damage_winning += all_statistics['red_winning_episode_red_enemy_damage']
            blue_ally_damage_losing += all_statistics['red_winning_episode_blue_ally_damage']
            blue_enemy_damage_losing += all_statistics['red_winning_episode_blue_enemy_damage']
        except:
            pass
        red_ally_damage_winning_mean.append(mean_statistics['red_winning_episode_red_ally_damage'])
        red_enemy_damage_winning_mean.append(mean_statistics['red_winning_episode_red_enemy_damage'])
        blue_ally_damage_losing_mean.append(mean_statistics['red_winning_episode_blue_ally_damage'])
        blue_enemy_damage_losing_mean.append(mean_statistics['red_winning_episode_blue_enemy_damage'])
        try:
            red_ally_kills += all_statistics['num_allies_killed_red']
            red_enemy_kills += all_statistics['num_enemies_killed_red']
            blue_ally_kills += all_statistics['num_allies_killed_blue']
            blue_enemy_kills += all_statistics['num_enemies_killed_blue']
        except:
            pass
        red_ally_kills_mean.append(mean_statistics['num_allies_killed_red'])
        red_enemy_kills_mean.append(mean_statistics['num_enemies_killed_red'])
        blue_ally_kills_mean.append(mean_statistics['num_allies_killed_blue'])
        blue_enemy_kills_mean.append(mean_statistics['num_enemies_killed_blue'])
        try:
            red_ally_kills_winning += all_statistics['red_winning_episode_red_ally_kills']
            red_enemy_kills_winning += all_statistics['red_winning_episode_red_enemy_kills']
            blue_ally_kills_losing += all_statistics['red_winning_episode_blue_ally_kills']
            blue_enemy_kills_losing += all_statistics['red_winning_episode_blue_enemy_kills']
        except:
            pass
        red_ally_kills_winning_mean.append(mean_statistics['red_winning_episode_red_ally_kills'])
        red_enemy_kills_winning_mean.append(mean_statistics['red_winning_episode_red_enemy_kills'])
        blue_ally_kills_losing_mean.append(mean_statistics['red_winning_episode_blue_ally_kills'])
        blue_enemy_kills_losing_mean.append(mean_statistics['red_winning_episode_blue_enemy_kills'])
        try:
            x_axis += [penalties[i]]*20
        except:
            pdb.set_trace()
            pass
        i += 1

red_inflicted_damage = red_enemy_damage_mean
red_taken_damage = np.add(np.asarray(blue_enemy_damage_mean), np.asarray(red_ally_damage_mean))
red_inflicted_damage_err = red_enemy_damage_std
red_taken_damage_err = max(blue_enemy_damage_std, red_ally_damage_std)
fig,(ax1)=plt.subplots(1,1)
for idx in range(len(red_inflicted_damage)):
    ax1.errorbar(x=red_taken_damage[idx], y=red_inflicted_damage[idx],
                 yerr=red_inflicted_damage_err[idx], xerr=red_taken_damage_err[idx],
                 label=penalties[idx], ecolor=colors[idx], ls='None')

plt.plot(np.linspace(0, 600, 100), np.linspace(0, 600, 100), linestyle='dashed')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.xlabel('Damage taken by red')
plt.ylabel('Damage inflicted by red')
plt.title('Damage')
plt.legend()
plt.show()
plt.savefig('compare.png')

'''
intentional_damage = red_enemy_damage_mean
unintentional_damage = np.add(np.asarray(blue_enemy_damage_mean), np.asarray(red_ally_damage_mean))
#unintentional_damage = red_ally_damage_mean[:5]
intentional_damage_err = red_enemy_damage_std
unintentional_damage_err = max(blue_enemy_damage_std, red_ally_damage_std)
#unintentional_damage_err = red_ally_damage_std[:5]
plt.errorbar(y=intentional_damage, x=unintentional_damage, yerr=intentional_damage_err, xerr=unintentional_damage_err,
             linestyle='None')
for i, (y,x) in enumerate(zip(intentional_damage, unintentional_damage)):
    plt.annotate(penalties[i], (x,y))
plt.plot(np.linspace(0, 600, 100), np.linspace(0, 600, 100), linestyle='dashed')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.xlabel('Damage taken by red')
plt.ylabel('Damage inflicted by red')
plt.title('Damage inflicted by policies')
plt.show()

intentional_damage = red_enemy_damage_mean[5:]
unintentional_damage = np.add(np.asarray(blue_enemy_damage_mean[5:]), np.asarray(red_ally_damage_mean[5:]))
#unintentional_damage = red_ally_damage_mean[5:]
intentional_damage_err = red_enemy_damage_std[5:]
unintentional_damage_err = max(blue_enemy_damage_std[5:], red_ally_damage_std[5:])
#unintentional_damage_err = red_ally_damage_std[5:]
plt.errorbar(y=intentional_damage, x=unintentional_damage, yerr=intentional_damage_err, xerr=unintentional_damage_err,
             linestyle='None')
for i, (y,x) in enumerate(zip(intentional_damage, unintentional_damage)):
    plt.annotate(penalties[i], (x,y))
plt.plot(np.linspace(0, 600, 100), np.linspace(0, 600, 100), linestyle='dashed')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.xlabel('Damage taken by red')
plt.ylabel('Damage inflicted by red')
plt.title('Damage inflicted by policies\n (trained only with friendly damage)')
plt.show()

plt.scatter(x_axis, blue_enemy_kills_losing)
plt.scatter(penalties, blue_enemy_kills_losing_mean, linewidths=2)
plt.xlim([0, 1])
plt.ylim([0, 5])
plt.xticks(penalties)
plt.xlabel('Penalty coefficient')
plt.ylabel('Kills')
plt.title('Kills from blue to red / Winning')
plt.show()

plt.errorbar(x=penalties, y=reward_mean, yerr=reward_std, linestyle='None', marker='^', elinewidth=0.5)
plt.xlim([0, 1])
plt.ylim([-5, 5])
plt.xticks(penalties)
plt.xlabel('Penalty coefficient')
plt.ylabel('Reward')
plt.title('Rewards Accumulated by Policies')
plt.show()
'''
