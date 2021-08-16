import math
import numpy as np
import torch
from sa import SA, DEFAULT_SCHEDULE

class Branin(object):

	def __init__(self):
		self.n_vertices = np.array([51, 51])
		
	def evaluate(self, x_g):
		flat = x_g.dim() == 1
		if flat:
			x_g = x_g.view(1, -1)
		ndim = x_g.size(1)
		assert ndim == len(self.n_vertices)
		n_repeat = int(ndim / 2)
		n_dummy = int(ndim % 2)

		x_e = torch.ones(x_g.size())
		for d in range(len(self.n_vertices)):
			x_e[:, d] = torch.linspace(-1, 1, int(self.n_vertices[d]))[x_g[:, d].long()]

		shift = torch.cat([torch.FloatTensor([2.5, 7.5]).repeat(n_repeat), torch.zeros(n_dummy)])

		x_e = x_e * 7.5 + shift

		a = 1
		b = 5.1 / (4 * math.pi ** 2)
		c = 5.0 / math.pi
		r = 6
		s = 10
		t = 1.0 / (8 * math.pi)
		output = 0
		for i in range(n_repeat):
			output += a * (x_e[:, 2 * i + 1] - b * x_e[:, 2 * i] ** 2 + c * x_e[:, 2 * i] - r) ** 2 \
			          + s * (1 - t) * torch.cos(x_e[:, 2 * i]) + s
		output /= float(n_repeat)
		if flat:
			return output.squeeze(0)
		else:
			return output

def run_branin(use_default_schedule=True):
	evaluator = Branin()
	init_state = [25,25]

	sa = SA(init_state, evaluator)
	if use_default_schedule:
		schedule = DEFAULT_SCHEDULE
	else:
		schedule = sa.auto(minutes=0.01, steps=100)
		schedule['steps'] = 100
	el = [] # list. (energy for each trial.)
	for i in range(25):
		sa = SA(init_state, evaluator)
		sa.copy_strategy = "slice"
		sa.set_schedule(schedule)
		state, e = sa.anneal()
		el.append(e)
	print("el :", el)
	print("avg : ",sum(el)/len(el))

if __name__ == '__main__':
	run_branin()