import numpy as np
import math
from sortedcontainers import SortedSet

class DataGenerator:
	def __init__(self, n=None, m=None, sparse=False, density=0.01):
		if (n is None) or (m is None):
			raise ValueError('Please, enter two dimensions')
		self.n = n
		self.m = m
		self.sparse = sparse
		self.density = density

	def generate_c(self):
		if self.sparse:
			num_nonzero = math.ceil(self.n * self.density)
			ind_nonzero = np.random.choice(range(self.n), size=num_nonzero, replace=False)
			val_nonzero = np.random.uniform(low=-5, high=5, size=num_nonzero)
			c = np.zeros(self.n)
			for i, ind in enumerate(ind_nonzero):
				c[ind] = val_nonzero[i]
		else:
			c = np.random.uniform(low=-5, high=5, size=self.n)
		return c

	def generate_A(self):
		A = []
		if self.sparse:
			for i in xrange(self.m):
				num_nonzero = math.ceil(self.n * self.density)
				ind_nonzero = np.random.choice(range(self.n), size=num_nonzero, replace=False)
				val_nonzero = np.random.uniform(low=0, high=5, size=num_nonzero)
				ai = np.zeros(self.n)
				for i, ind in enumerate(ind_nonzero):
					ai[ind] = val_nonzero[i]
				A.append(ai)
		else:
			for i in xrange(self.m):
				ai = np.random.uniform(size=self.n)
				A.append(ai)
		A = np.vstack(A)
		return A

	def generate_b(self):
		b = np.random.uniform(low=0, high=5, size=self.m)
		return b


class _StochasticOracle:
	def __init__(self, c, A, b):
		self.c = c
		self.A = A
		self.b = b
		self.m = A.shape[0]
		self.n = A.shape[1]
		if np.min(c) < 0:
			self.min_c = np.min(c)
		else:
			self.min_c = 0.0

	def initialize(self, x):
		self.nonzero_as = {} #dict n_component : 'a's which have this component nonzero
		for i_col, col in enumerate(self.A.T):
			self.nonzero_as[i_col] = np.nonzero(col)
		list_of_products = [np.dot(ai, x) - bi for ai, bi in zip(self.A, self.b)]
		self.products_set = SortedSet(list_of_products)
		self.components_products = dict(enumerate(list_of_products))
		self.products_components = dict(zip(list_of_products, range(self.m)))

	def answer(self):
		sum_c = sum(self.c - self.min_c)
		c_ind = np.random.choice(range(self.n), size=1, p=((self.c - self.min_c)/sum_c))[0]
		df = np.zeros(self.n)
		#df[c_ind] = 1
		df[c_ind] = sum_c * self.c[c_ind]
		g = self.products_set[-1] #max value in the constrains
		ind_max = self.products_components[g] #index of max value in the constrains
		a = self.A[ind_max] #vector a from A which gives the max
		sum_a = sum(a)
		dg_ind = np.random.choice(range(self.n), size=1, p=a/sum_a)[0]
		dg = np.zeros(self.n)
		#dg[dg_ind] = 1
		dg[dg_ind] = sum_a * a[dg_ind]
		return df, dg, g, ind_max, c_ind, dg_ind

	def update(self, x, changed_component):
		i_vectors = self.nonzero_as[changed_component][0] #array of indices of vectors with nonzero changed_component
		for i_vector in i_vectors:
			val_vector = self.components_products[i_vector]
			self.products_components.pop(val_vector) #remove item from dict
			self.products_set.remove(val_vector) #remove a value from 'heap'
			new_val = np.dot(self.A[i_vector], x) - self.b[i_vector]
			self.products_set.add(new_val) #add new value to 'heap'
			self.components_products[i_vector] = new_val
			self.products_components[new_val] = i_vector



class Mirror:
	def  __init__(self, c=None, A=None, b=None):
		if (c is None) or (A is None) or (b is None):
			raise ValueError('Please, enter all data')
		self.c = c
		self.A = A
		self.b = b
		self.m = A.shape[0]
		self.n = A.shape[1]
		self.Mf = np.linalg.norm(c)
		self.Mg = np.max(np.linalg.norm(A, axis=1))


	def objective(self, x):
		return np.dot(self.c, x)


	def dual(self, x):
		return -np.dot(self.b, x)


	def _deterministic_oracle(self, x):
		df = self.c
		constrains_val = np.array([np.dot(ai, x) - bi for ai, bi in zip(self.A, self.b)])
		ind_max = np.argmax(constrains_val)
		dg = self.A[ind_max]
		g = constrains_val[ind_max]
		return df, dg, g, ind_max


	def _accuracy_solve(self, eps, stochastic, check_every, max_iter, trace):
		hf = eps / (self.Mf * self.Mg)
		hg = eps / (self.Mg)**2
		eps_g = eps
		eps_f = (self.Mf / self.Mg) * eps_g
		I = 0 #count number of inner updates
		J = [] #store indices of max in constrains
		xI = [] #store inner points
		tr = {} #if trace
		tr['objective'] = []
		tr['constraints'] = []
		tr['gap'] = []
		iter_count = 0 #count number of iterations
		x = np.zeros(self.n)
		gap = 10000.0
		if stochastic:
			so = _StochasticOracle(self.c, self.A, self.b)
			so.initialize(x)
		while gap > eps_f:
			if stochastic:
				df, dg, g, ind_max, c_ind, dg_ind = so.answer()
			else:
				df, dg, g, ind_max = self._deterministic_oracle(x)
			if trace:
				tr['objective'].append(self.objective(x))
				tr['constraints'].append(g)
			if g <= eps:
				xI.append(x)
				I += 1
				x = x - hf * df
				x[x < 0] = 0
				if stochastic:
					so.update(x, c_ind)
			else:
				J.append(ind_max)
				x = x - hg * dg
				x[x < 0] = 0
				if stochastic:
					so.update(x, dg_ind)
			iter_count += 1
			if iter_count >= max_iter:
				print 'norm(x) = ' + str(np.linalg.norm(x))
				print 'Maximum number of iterations reached'
				break
			if iter_count % check_every == 0:
				try:
					xN = sum(xI) / I
				except ZeroDivisionError:
					print 'Interrupt: set bigger check_every parameter'
				lam = np.zeros(self.m)
				for ind_max in J:
					lam[ind_max] += 1
				lam = lam * hg / (hf * I)
				gap = self.objective(xN) - self.dual(lam)
				tr['gap'].append(gap)
		try:
			xN = sum(xI) / I
		except ZeroDivisionError:
			print 'Algorithm has not converged'
		lam = np.zeros(self.m)
		for ind_max in J:
			lam[ind_max] += 1
		if trace:
			return xN, lam, tr
		else:
			return xN, lam


	def _iter_solve(self, eps, N_iter, stochastic, trace):
		hf = eps / (self.Mf * self.Mg)
		hg = eps / (self.Mg)**2
		I = 0 #count number of inner updates
		J = [] #store indices of max in constrains
		xI = [] #store inner points
		tr = {} #if trace
		tr['objective'] = []
		tr['constraints'] = []
		tr['grad_objective'] = []
		tr['grad_constraints'] = []
		x = np.zeros(self.n)
		if stochastic:
			so = _StochasticOracle(self.c, self.A, self.b)
			so.initialize(x)
		for it in xrange(N_iter):
			if stochastic:
				df, dg, g, ind_max, c_ind, dg_ind = so.answer()
			else:
				df, dg, g, ind_max = self._deterministic_oracle(x)
			if trace:
				tr['objective'].append(self.objective(x))
				tr['constraints'].append(g)
				tr['grad_objective'].append(df)
				tr['grad_constraints'].append(dg)
			if g <= eps:
				xI.append(x)
				I += 1
				x = x - hf * df
				#print x
				x[x < 0] = 0 #projection
				#print x
				if stochastic:
					so.update(x, c_ind)
			else:
				J.append(ind_max)
				x = x - hg * dg
				x[x < 0] = 0
				if stochastic:
					so.update(x, dg_ind)
		try:
			xN = sum(xI) / I
		except ZeroDivisionError:
			print 'Algorithm has not converged'
			if trace:
				return None, None, tr
			else:
				return None, None
		lam = np.zeros(self.m)
		for ind_max in J:
			lam[ind_max] += 1
		lam = lam * hg / (hf * I)
		if trace:
			return xN, lam, tr
		else:
			return xN, lam


	def solve(self, eps=0.1, N_iter=None, stochastic=False, check_every=None, max_iter=100000, trace=False):
		if N_iter is None:
			return self._accuracy_solve(eps, stochastic, check_every, max_iter, trace)
			
		else:
			return self._iter_solve(eps, N_iter, stochastic, trace)
