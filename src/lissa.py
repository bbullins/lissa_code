import numpy as np
from logistic_oracles import *
import random, time 

def lissa_main(init_x, num_epochs, num_lissa_iter, outer_grad_size, hessian_batch_size, quad_stepsize, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x

	epochs = []
	wall_times = []
	trainerror = []
	
	num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
	start_time = time.time()

	for curr_iter in range(int(num_iter)):		
		epochs += [curr_iter*(outer_grad_size + num_lissa_iter*hessian_batch_size)/num_examples]
		wall_times += [time.time() - start_time]
		trainerror += [data_holder.logistic_batch_func(range(0, num_examples), curr_x)]
		curr_grad = data_holder.logistic_batch_grad(range(0, num_examples), curr_x)
		curr_step = np.zeros(curr_x.size)
		
		for lissa_iter in range(num_lissa_iter):
			rand_index = random.sample(xrange(num_examples), hessian_batch_size)
			sub_step = data_holder.logistic_batch_hess_vec_product(rand_index, curr_x, curr_step)
			curr_quad_step = curr_grad - sub_step
			
			curr_step = curr_step + quad_stepsize*curr_quad_step

		curr_x = curr_x - curr_step

	output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}	
	return curr_x, output_data

def grad_descent(num_iter, init_x, stepsize, batch_size, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x
	
	for curr_iter in range(num_iter):
		curr_grad = data_holder.logistic_batch_grad(random.sample(xrange(num_examples), batch_size), curr_x)
		curr_x = curr_x - stepsize*curr_grad

	return curr_x

