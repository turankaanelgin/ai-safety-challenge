import os, json
import pdb

import numpy as np


def take_first(elem):
    return elem[0]

main_eval_folder = './logs/curriculum'
for policy_folder in os.listdir(main_eval_folder):
    if policy_folder.endswith('json'):
        continue
    if not policy_folder.endswith('EVAL'):
        continue

    red_red_damages = []
    red_blue_damages = []
    blue_red_damages = []
    red_red_damages_mean = []
    red_blue_damages_mean = []
    blue_red_damages_mean = []
    difference_mean = []

    for seed_folder in os.listdir(os.path.join(main_eval_folder, policy_folder, '999999')):
        file_path = os.path.join(main_eval_folder, policy_folder, '999999', seed_folder, 'mean_statistics_per_env.json')
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            all_statistics = json.load(f)

        red_red_damages += all_statistics['All-Red-Red-Damage']
        red_blue_damages += all_statistics['All-Red-Blue Damage']
        blue_red_damages += all_statistics['All-Blue-Red Damage']

        mean_red_red = np.mean(all_statistics['All-Red-Red-Damage'])
        mean_red_blue = np.mean(all_statistics['All-Red-Blue Damage'])
        mean_blue_red = np.mean(all_statistics['All-Blue-Red Damage'])
        red_red_damages_mean.append(mean_red_red)
        red_blue_damages_mean.append(mean_red_blue)
        blue_red_damages_mean.append(mean_blue_red)

        difference = mean_red_blue - (mean_red_red + mean_blue_red)
        difference_mean.append((difference, (mean_red_blue, mean_red_red, mean_blue_red)))

    if len(difference_mean) < 6:
        continue

    total_mean_red_red = np.mean(red_red_damages_mean)
    total_mean_red_blue = np.mean(red_blue_damages_mean)
    total_mean_blue_red = np.mean(blue_red_damages_mean)
    total_median_red_red = np.median(red_red_damages_mean)
    total_median_red_blue = np.median(red_blue_damages_mean)
    total_median_blue_red = np.median(blue_red_damages_mean)
    total_std_red_red = np.std(red_red_damages)
    total_std_red_blue = np.std(red_blue_damages)
    total_std_blue_red = np.std(blue_red_damages)
    difference_mean.sort(key=take_first)
    total_max_red_blue, total_max_red_red, total_max_blue_red = difference_mean[-1][1]

    stats_dict = {'Mean Red-Red Damage': total_mean_red_red,
                  'Mean Red-Blue Damage': total_mean_red_blue,
                  'Mean Blue-Red Damage': total_mean_blue_red,
                  'Median Red-Red Damage': total_median_red_red,
                  'Median Red-Blue Damage': total_median_red_blue,
                  'Median Blue-Red Damage': total_median_blue_red,
                  'Std Red-Red Damage': total_std_red_red,
                  'Std Red-Blue Damage': total_std_red_blue,
                  'Std Blue-Red Damage': total_std_blue_red,
                  'Max Red-Red Damage': total_max_red_red,
                  'Max Red-Blue Damage': total_max_red_blue,
                  'Max Blue-Red Damage': total_max_blue_red,}
    with open(os.path.join(main_eval_folder, policy_folder, '999999', 'mean_statistics_per_env.json'), 'w+') as f:
        json.dump(stats_dict, f, indent=4)
