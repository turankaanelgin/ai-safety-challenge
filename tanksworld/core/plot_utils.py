# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import pdb

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import random, colorsys

def plot_policy_records_damage(records, windows, alphas, filename, colors=None, offsets=None,
							   episodic=False, fig=None, ax=None, return_figure=False):

	default_colors = []
	hues = [1, 4, 7, 10, 3, 6, 9, 12, 2, 5, 8, 11]
	for h in hues:
		default_colors.append(randomRGBPure(float(h) / 12.0))

	colors = default_colors[:3]

	if offsets is None:
		offsets = []
		for _ in records:
			offsets.append(0)

	if fig is None:
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))

	for widx, windowsize in enumerate(windows[:1]):

		alpha = alphas[-1]

		for pridx, pol in enumerate(records):

			ylabel = 'Episodic Damage'

			red_blue_damages = pol.ep_red_blue_damages
			red_red_damages = pol.ep_red_red_damages
			blue_red_damages = pol.ep_blue_red_damages

			rolling_avg_buffer_red_blue = []
			rolling_avg_buffer_red_red = []
			rolling_avg_buffer_blue_red = []
			rolling_avg_damages_red_red = []
			rolling_avg_damages_red_blue = []
			rolling_avg_damages_blue_red = []
			rolling_avg_steps = []

			for i in range(len(red_blue_damages)):
				rolling_avg_buffer_red_blue.append(red_blue_damages[i])
				rolling_avg_buffer_red_red.append(red_red_damages[i])
				rolling_avg_buffer_blue_red.append(blue_red_damages[i])
				if len(rolling_avg_buffer_red_red) > windowsize:
					rolling_avg_buffer_red_red.pop(0)
					rolling_avg_buffer_red_blue.pop(0)
					rolling_avg_buffer_blue_red.pop(0)
					rolling_avg_damages_red_red.append(np.mean(rolling_avg_buffer_red_red))
					rolling_avg_damages_red_blue.append(np.mean(rolling_avg_buffer_red_blue))
					rolling_avg_damages_blue_red.append(np.mean(rolling_avg_buffer_blue_red))

					if episodic:
						rolling_avg_steps.append(i + 1)

			# plot
			#if widx == len(windows)-1:
			ax.plot(rolling_avg_steps, rolling_avg_damages_red_blue, color=colors[0], alpha=alpha, label='Red-Blue')
			ax.plot(rolling_avg_steps, rolling_avg_damages_red_red, color=colors[1], alpha=alpha, label='Red-Red')
			ax.plot(rolling_avg_steps, rolling_avg_damages_blue_red, color=colors[2], alpha=alpha, label='Blue-Red')

	if episodic:
		ax.set_xlabel("Episodes")
	else:
		ax.set_xlabel("Steps")
	ax.set_ylabel(ylabel)
	ax.legend()

	# change .png to _steps.png or _episodes.png
	if ".png" not in filename:
		filename = filename + ".png"

	if episodic:
		filename = filename.replace(".png", "_episodes.png")
	else:
		filename = filename.replace(".png", "_steps.png")

	if return_figure:
		return fig, ax
	else:
		plt.savefig(filename)
		# clear the figure so we dont plot on top of other plots
		plt.close(fig)

'''
def plot_policy_records_damage(records, windows, alphas, filename, colors=None, offsets=None,
	episodic=False, fig=None, ax=None, return_figure=False):

	default_colors = []
	hues = [1, 4, 7, 10, 3, 6, 9, 12, 2, 5, 8, 11]
	for h in hues:
		default_colors.append(randomRGBPure(float(h) / 12.0))

	colors = default_colors[:3]

	if offsets is None:
		offsets = []
		for _ in records:
			offsets.append(0)

	if fig is None:
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))

	for widx, windowsize in enumerate(windows):

		alpha = alphas[widx]

		for pridx, pol in enumerate(records):

			ylabel = pol.ylabel

			steps = pol.ep_cumlens
			red_blue_damages = pol.ep_red_blue_damages
			red_red_damages = pol.ep_red_red_damages
			blue_red_damages = pol.ep_blue_red_damages

			for k in range(len(steps)):
				steps[k] += offsets[pridx]

			rolling_avg_buffer_red_blue = []
			rolling_avg_buffer_red_red = []
			rolling_avg_buffer_blue_red = []
			rolling_avg_damages_red_red = []
			rolling_avg_damages_red_blue = []
			rolling_avg_damages_blue_red = []
			rolling_avg_steps = []

			for i in range(len(steps)):
				rolling_avg_buffer_red_blue.append(red_blue_damages[i])
				rolling_avg_buffer_red_red.append(red_red_damages[i])
				rolling_avg_buffer_blue_red.append(blue_red_damages[i])
				if len(rolling_avg_buffer_red_red) > windowsize:
					rolling_avg_buffer_red_red.pop(0)
					rolling_avg_buffer_red_blue.pop(0)
					rolling_avg_buffer_blue_red.pop(0)
					rolling_avg_damages_red_red.append(np.mean(rolling_avg_buffer_red_red))
					rolling_avg_damages_red_blue.append(np.mean(rolling_avg_buffer_red_blue))
					rolling_avg_damages_blue_red.append(np.mean(rolling_avg_buffer_blue_red))

					if episodic:
						rolling_avg_steps.append(i + 1)
					else:
						rolling_avg_steps.append(steps[i])

			# plot
			if widx == len(windows)-1:
				ax.plot(rolling_avg_steps, rolling_avg_damages_red_blue, color=colors[0], alpha=alpha, label='Red-Blue')
				ax.plot(rolling_avg_steps, rolling_avg_damages_red_red, color=colors[1], alpha=alpha, label='Red-Red')
				ax.plot(rolling_avg_steps, rolling_avg_damages_blue_red, color=colors[2], alpha=alpha, label='Blue-Red')

	if episodic:
		ax.set_xlabel("Episodes")
	else:
		ax.set_xlabel("Steps")
	ax.set_ylabel(ylabel)
	ax.legend()

	# change .png to _steps.png or _episodes.png
	if ".png" not in filename:
		filename = filename + ".png"

	if episodic:
		filename = filename.replace(".png", "_episodes.png")
	else:
		filename = filename.replace(".png", "_steps.png")

	if return_figure:
		return fig, ax
	else:
		plt.savefig(filename)
		# clear the figure so we dont plot on top of other plots
		plt.close(fig)
'''


def plot_policy_records_std(records, windows, alphas, filename, colors=None, offsets=None,
							episodic=False, fig=None, ax=None, return_figure=False):

	default_colors = []
	hues = [1, 4, 7, 10, 3, 6, 9, 12, 2, 5, 8, 11]
	for h in hues:
		default_colors.append(randomRGBPure(float(h) / 12.0))

	colors = default_colors[:3]

	if offsets is None:
		offsets = []
		for _ in records:
			offsets.append(0)

	if fig is None:
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))

	for widx, windowsize in enumerate(windows):

		alpha = alphas[widx]

		for pridx, pol in enumerate(records):

			ylabel = pol.ylabel

			steps = pol.ep_cumlens
			ep_stds = pol.ep_stds

			for k in range(len(steps)):
				steps[k] += offsets[pridx]

			rolling_avg_buffer = [[], [], []]
			rolling_avg_stds = [[], [], []]
			rolling_avg_steps = []

			for i in range(len(steps)):
				rolling_avg_buffer[0].append(ep_stds[0][i])
				rolling_avg_buffer[1].append(ep_stds[1][i])
				rolling_avg_buffer[2].append(ep_stds[2][i])

				if len(rolling_avg_buffer[0]) > windowsize:
					rolling_avg_buffer[0].pop(0)
					rolling_avg_buffer[1].pop(0)
					rolling_avg_buffer[2].pop(0)
					rolling_avg_stds[0].append(np.mean(rolling_avg_buffer[0]))
					rolling_avg_stds[1].append(np.mean(rolling_avg_buffer[1]))
					rolling_avg_stds[2].append(np.mean(rolling_avg_buffer[2]))

					if episodic:
						rolling_avg_steps.append(i + 1)
					else:
						rolling_avg_steps.append(steps[i])

			# plot
			ax.plot(rolling_avg_steps, rolling_avg_stds[0], color=colors[0], alpha=alpha, label='Translate')
			ax.plot(rolling_avg_steps, rolling_avg_stds[1], color=colors[1], alpha=alpha, label='Orient')
			ax.plot(rolling_avg_steps, rolling_avg_stds[2], color=colors[2], alpha=alpha, label='Shoot')

	if episodic:
		ax.set_xlabel("Episodes")
	else:
		ax.set_xlabel("Steps")
	ax.set_ylabel(ylabel)
	ax.legend()

	# change .png to _steps.png or _episodes.png
	if ".png" not in filename:
		filename = filename + ".png"

	if episodic:
		filename = filename.replace(".png", "_episodes.png")
	else:
		filename = filename.replace(".png", "_steps.png")

	if return_figure:
		return fig, ax
	else:
		plt.savefig(filename)
		# clear the figure so we dont plot on top of other plots
		plt.close(fig)


#plot episode results from policies, 
#averaging over windows and displaying with alphas
#saving to filename
def plot_policy_records(records, windows, alphas, filename, colors=None, offsets=None, 
	episodic=False, fig=None, ax=None, return_figure=False, intrinsic=False):
	
	# get the main channels if these are record objects
	if hasattr(records[0], "channels"):
		records = [r.channels["main"] for r in records]

	default_colors = []
	hues = [1, 4, 7, 10, 3, 6, 9, 12, 2, 5, 8, 11]
	for h in hues:
		default_colors.append(randomRGBPure(float(h)/12.0))

	if colors is None:
		colors = []
		idx = 0
		for pr in records:
			colors.append(default_colors[idx])
			idx += 1
			if idx >= len(default_colors):
				idx = 0

	if offsets is None:
		offsets = []
		for pr in records:
			offsets.append(0)

	if fig is None:
		fig, ax = plt.subplots(1,1,figsize=(12,6))

	for widx, windowsize in enumerate(windows):

		alpha = alphas[widx]

		for pridx, pol in enumerate(records):

			ylabel = 'Episodic Reward'

			steps = pol.ep_cumlens
			if intrinsic:
				results = pol.ep_intrinsic
			else:
				results = pol.ep_results

			for k in range(len(steps)):
				steps[k] += offsets[pridx]

			rolling_avg_buffer = []
			rolling_avg_results = []
			rolling_avg_steps = []

			for i in range(len(steps)):
				rolling_avg_buffer.append(results[i])

				if len(rolling_avg_buffer) > windowsize:
					rolling_avg_buffer.pop(0)
					rolling_avg_results.append(np.mean(rolling_avg_buffer))

					if episodic:
						rolling_avg_steps.append(i+1)
					else:
						rolling_avg_steps.append(steps[i])

			#plot
			ax.plot(rolling_avg_steps, rolling_avg_results, color=colors[pridx], alpha=alpha)

	if episodic:
		ax.set_xlabel("Episodes")
	else:
		ax.set_xlabel("Steps")
	ax.set_ylabel(ylabel)

	# change .png to _steps.png or _episodes.png
	if ".png" not in filename:
		filename = filename + ".png"

	if episodic:
		filename = filename.replace(".png", "_episodes.png")
	else:
		filename = filename.replace(".png", "_steps.png")

	if return_figure:
		return fig, ax
	else:
		plt.savefig(filename)
		#clear the figure so we dont plot on top of other plots
		plt.close(fig)


def randomRGBPure(hue=None):
    h = random.random() if hue is None else hue
    s = random.uniform(0.8,0.9)
    v = random.uniform(0.8,0.9)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r,g,b)