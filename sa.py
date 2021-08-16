from simanneal import Annealer
import random
import torch
import numpy as np

DEFAULT_SCHEDULE = {'tmax' : 25000.0, 'tmin' : 2.5, 'steps' : 320, 'updates' : 100}

class SA(Annealer):
	def __init__(self, init_state, evaluator):
		self.n_vertices = evaluator.n_vertices
		super(SA, self).__init__(init_state)  
		self.evaluator = evaluator

	def move(self):
		initial_energy = self.energy()
		
		# randomly choose an index to move
		lidx = random.randint(0, len(self.n_vertices)-1)
		nv = self.n_vertices[lidx]
		val = self.state[lidx]

		# randomly choose category different to current one.
		while val == self.state[lidx]:
			val = random.randint(0, nv-1)
		self.state[lidx] = val
		return self.energy() - initial_energy

	def energy(self):
		energy = self.evaluator.evaluate(torch.from_numpy(np.array(self.state))).item()
		return energy
    
def sample_init_points(n_vertices, n_points, random_seed=None):
    """

    :param n_vertices: 1D np.array
    :param n_points:
    :param random_seed:
    :return:
    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat([init_points, torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1)], dim=0)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points
