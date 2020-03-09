

from tanksworld.env import TanksWorldEnv
import my_config as cfg

# example of passing in kwargs and using them
def make_env(**kwargs):
	return TanksWorldEnv(cfg.args.exe,
		action_repeat=6, 			# step between decisions, will be 6 in evaluation
		image_scale=128,            # image size, will be 128 in evaluation
		timeout=500,				# maximum number of steps before episode forces a reset
		friendly_fire=True, 		# do you get penalized for damaging self, allies, neutral
		take_damage_penalty=True,   # do you get penalized for receiving damage (double counts w/ self-freindly-fire)
		kill_bonus=True, 			# do you get +1 for killing enemy (-1 penalty for friendly fire kills if friendly fire is on)
		death_penalty=True,			# do you get -1 for dying
        static_tanks=kwargs["static_tanks"], 			# indices of tanks that do not move (not exposed externally, changes number of controllable players)
        random_tanks=kwargs["random_tanks"],	# indices of tanks that move randomly (not exposed externally, changes number of controllable players)
        disable_shooting=kwargs["disable_shooting"], 		# indices of tanks that cannot shoot (i.e. to allow random movement without shooting)
        will_render=True)			# prepare rgb images for displaying when render() is called.  If not rendering turn off.

		# NOTE: Make sure if you set static_tanks or random_tanks, or adjuist image_scale, that you make appropriate changes in my_config.py!!!