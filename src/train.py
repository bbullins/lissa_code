from logistic_oracles import *
from lissa import *
import numpy as np
import cPickle
import argparse

data_holder = DataHolder(lam=1e-4, dataset = 'MNIST')
num_examples = data_holder.num_train_examples

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=25, type=int)
parser.add_argument('--num_lissa_iter', default=num_examples, type=int)
parser.add_argument('--outer_grad_size', default=num_examples, type=int)
parser.add_argument('--hessian_batch_size', default=1, type=int)
parser.add_argument('--stepsize', default=1.0, type=float)
args = parser.parse_args()

gd_init_x = np.zeros(data_holder.data_dim)
gd_iter = 5
gd_stepsize = 5.0
init_x = grad_descent(gd_iter, gd_init_x, gd_stepsize, num_examples, data_holder)

num_epochs = args.num_epochs
num_lissa_iter = args.num_lissa_iter
outer_grad_size = args.outer_grad_size
hessian_batch_size = args.hessian_batch_size
stepsize = args.stepsize

print '-----------------------------------------------'
print 'Training model...'
print '-----------------------------------------------\n'

f = open('../output/LiSSAOutputMNIST','wb')
result, output_data = lissa_main(init_x, num_epochs, num_lissa_iter, outer_grad_size, hessian_batch_size, stepsize, data_holder)
cPickle.dump(output_data, f)
f.close()

print '-----------------------------------------------'
print 'Training complete'
print '-----------------------------------------------'
