import numpy as np
import cPickle, gzip, math
from sklearn.preprocessing import normalize

class DataHolder:
	def __init__(self, dataset = 'MNIST', lam = 0):
		self.lam = lam
		self.load_dataset(dataset)
	
	
	def load_dataset(self, dataset):
		if dataset == 'MNIST':
			print '-----------------------------------------------'
			print 'Loading MNIST 4/9 data...'
			print '-----------------------------------------------\n'
			self.load_mnist_49()
		else:
			raise ValueError('No Dataset exists by that name')
		
		self.data_dim = self.train_set[0][0].size
		self.num_train_examples = self.train_set[0].shape[0]
		self.num_test_examples = self.test_set[0].shape[0]

	### The following functions implement the logistic function oracles
	
	## This is to load the 49 mnist
	def load_mnist_49(self):
		f = open('../data/mnist49data', 'rb')
		train_set, valid_set, test_set = cPickle.load(f)
		f.close()
		self.train_set = [normalize(train_set[0], axis = 1, norm = 'l2'), train_set[1]]
		self.valid_set = [normalize(valid_set[0], axis = 1, norm = 'l2'), valid_set[1]]
		self.test_set = [normalize(test_set[0], axis = 1, norm = 'l2'), test_set[1]]


	## First implement the function for individuals
	
	def fetch_correct_datamode(self, mode = 'TRAIN'):
		if mode == 'TRAIN':
			return self.train_set
		elif mode == 'VALIDATE':
			return self.validate_set
		elif mode == 'TEST':
			return self.test_set
		else:
			raise ValueError('Wrong mode value provided')
	
	def logistic_indiv_func(self, data_index, model, mode='TRAIN'):
		data_set = self.fetch_correct_datamode(mode)
		v = -1.0*data_set[1][data_index]*np.dot(model, data_set[0][data_index])
		return np.log(np.exp(v) + 1)
	
	def logistic_indiv_grad(self, data_index, model):
		data_set = self.train_set
		v = -1.0*data_set[1][data_index]*np.dot(model, data_set[0][data_index])
		return -1*data_set[1][data_index]*data_set[0][data_index]*(np.exp(v)/(1 + np.exp(v)))

	def logistic_indiv_grad_coeff(self, data_index, model):
		data_set = self.train_set
		v = -1.0*data_set[1][data_index]*np.dot(model, data_set[0][data_index])
		return -1*data_set[1][data_index]*(np.exp(v)/(1 + np.exp(v)))
	
	def logistic_indiv_hess(self, data_index, model):
		data_set = self.train_set
		v = -1.0*data_set[1][data_index]*np.dot(model, data_set[0][data_index])
		return (data_set[1][data_index])*((math.pow(np.exp(v),0.5))/(np.exp(v)+1))*data_set[0][data_index]

	def logistic_batch_func(self, data_batch, model):
		func_val = 0.0
		for data_indiv in data_batch:
			func_val += self.logistic_indiv_func(data_indiv, model, 'TRAIN')
		avg_func_val = func_val / len(data_batch)
		return avg_func_val + self.lam*np.dot(model, model)

	def logistic_batch_grad(self, data_batch, model):
		batch_grad = np.zeros(self.data_dim)
		for data_indiv in data_batch:
			batch_grad += self.logistic_indiv_grad(data_indiv, model)
		avg_batch_grad = batch_grad / len(data_batch)
		return avg_batch_grad + 2*self.lam*model

	def logistic_batch_hess_full(self, data_batch, model):
		batch_hess = np.zeros((self.data_dim, self.data_dim))
		for data_indiv in data_batch:
			batch_hess += np.outer(self.logistic_indiv_hess(data_indiv, model), self.logistic_indiv_hess(data_indiv, model))
		avg_batch_hess = batch_hess / len(data_batch)
		return avg_batch_hess + 2*self.lam*np.identity(self.data_dim)

	def logistic_batch_hess_vec_product(self, data_batch, model, vector):
		hess_vec = np.zeros(self.data_dim)
		for data_indiv in data_batch:
			vtemp = self.logistic_indiv_hess(data_indiv, model)
			hess_vec += np.dot(vtemp, vector)*vtemp
		
		avg_hess_vec = hess_vec/len(data_batch)
		return avg_hess_vec + 2*self.lam*vector
	
	def test_error(self, model):
		func_val = 0.0
		data_batch = range(0, self.num_test_examples)
		for data_indiv in data_batch:
			func_val += self.logistic_indiv_func(data_indiv, model, 'TEST')
		avg_func_val = func_val / len(data_batch)
		return avg_func_val
	
	def error_01(self, model, mode='TRAIN'):
		data_set = self.fetch_correct_datamode(mode)
		num_examples = data_set[0].shape[0]
		error = 0
		for i in range(num_examples):
			error += abs(np.sign(np.dot(model, data_set[0][i])) - data_set[1][i])/2

		return error/num_examples

	