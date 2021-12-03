import pdb

import numpy as np
import json
import os
from collections import defaultdict


red_enemy_damage = defaultdict(int)
red_ally_damage = defaultdict(int)
blue_enemy_damage = defaultdict(int)
for folder in os.listdir('./logs/final-baseline'):
    if folder == 'seeds.json' or folder == 'overall_metrics.json':
        continue
    config = folder.split('____')[0]
    with open(os.path.join('./logs/final-baseline', folder, 'mean_statistics.json'), 'r') as f:
        mean_stats = json.load(f)
    red_enemy_damage[config] += mean_stats['red_enemy_damage']
    red_ally_damage[config] += mean_stats['red_ally_damage']
    blue_enemy_damage[config] += mean_stats['blue_enemy_damage']

for config in red_enemy_damage:
    red_enemy_damage[config] /= 5
    red_ally_damage[config] /= 5
    blue_enemy_damage[config] /= 5

dict = {'red_enemy_damage': red_enemy_damage,
        'red_ally_damage': red_ally_damage,
        'blue_enemy_damage': blue_enemy_damage}
with open('./logs/final-baseline/overall_metrics.json', 'w+') as f:
    json.dump(dict, f, indent=4)