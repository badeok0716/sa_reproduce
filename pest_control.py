import numpy as np
import sys, os
import torch
from sa import SA, DEFAULT_SCHEDULE, sample_init_points
PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25
class PestControl(object):
	"""
	Ising Sparsification Problem with the simplest graph
	"""
	def __init__(self, random_seed=None):
		self.n_vertices = np.array([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES)

	def evaluate(self, x):
		assert x.numel() == len(self.n_vertices)
		if x.dim() == 2:
			x = x.squeeze(0)
		evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy())
		return float(evaluation) * x.new_ones((1,)).float()


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
	if apply_control:
		next_pest_frac = (1.0 - control_rate) * curr_pest_frac
	else:
		next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
	return next_pest_frac


def _pest_control_score(x):
	U = 0.1
	n_stages = x.size
	n_simulations = 100

	init_pest_frac_alpha = 1.0
	init_pest_frac_beta = 30.0
	spread_alpha = 1.0
	spread_beta = 17.0 / 3.0

	control_alpha = 1.0
	control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
	tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
	control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
	# below two changes over stages according to x
	control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

	payed_price_sum = 0
	above_threshold = 0

	init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
	curr_pest_frac = init_pest_frac
	for i in range(n_stages):
		spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
		do_control = x[i] > 0
		if do_control:
			control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
			next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
			# torelance has been developed for pesticide type 1
			control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
			# you will get discount
			payed_price = control_price[x[i]] * (1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
		else:
			next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
			payed_price = 0
		payed_price_sum += payed_price
		above_threshold += np.mean(curr_pest_frac > U)
		curr_pest_frac = next_pest_frac

	return payed_price_sum + above_threshold

def run_pestcontrol(use_default_schedule=True):
	evaluator = PestControl(0)
	init_state = sample_init_points(evaluator.n_vertices, 1, 0).long().numpy()[0]
	sa = SA(init_state, evaluator)	
	if use_default_schedule:
		schedule = DEFAULT_SCHEDULE
	else:
		schedule = sa.auto(minutes=0.01, steps=320)
		schedule['steps'] = 320

	el = [] # list. (energy for each trial.)
	for i in range(25):
		init_state = sample_init_points(evaluator.n_vertices, 1, i).long().numpy()[0]
		sa = SA(init_state, evaluator)
		sa.set_schedule(schedule)
		state, e = sa.anneal()
		el.append(e)
	print("avg : ",sum(el)/len(el))

if __name__ == '__main__':
	run_pestcontrol()