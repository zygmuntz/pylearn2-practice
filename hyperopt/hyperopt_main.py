#!/usr/bin/env python

import numpy as np
import sys

from pylearn2 import utils
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.space import VectorSpace
from pylearn2.utils import serial

###

train_file = 'data/train.csv'
validation_file = 'data/validation.csv'
output_model_file = 'saved_clf.pkl'

###

def get_error_from_model( model_path ):

	model = serial.load( model_path )
	error = min( model.monitor.channels['valid_objective'].val_record )
	return error

def main( x ):

	l1_dim = x[0]
	l2_dim = x[1]
	learning_rate = x[2]
	momentum = x[3]
	l1_dropout = x[4]
	decay_factor = x[5]
	
	min_lr = 1e-7

	#

	train = np.loadtxt( train_file, delimiter = ',' )
	x_train = train[:,0:-1]
	y_train = train[:,-1]
	y_train.shape = ( y_train.shape[0], 1 )

	# 

	validation = np.loadtxt( validation_file, delimiter = ',' )
	x_valid = validation[:,0:-1]
	y_valid = validation[:,-1]
	y_valid.shape = ( y_valid.shape[0], 1 )

	#

	#input_space = VectorSpace( dim = x.shape[1] )
	full = DenseDesignMatrix( X = x_train, y = y_train )
	valid = DenseDesignMatrix( X = x_valid, y = y_valid )

	l1 = mlp.RectifiedLinear( 
		layer_name='l1',
		irange=.001,
		dim = l1_dim,
		# "Rather than using weight decay, we constrain the norms of the weight vectors"
		max_col_norm=1.
	)

	l2 = mlp.RectifiedLinear(
		layer_name='l2',
		irange=.001,
		dim = l2_dim,
		max_col_norm=1.
	)

	output = mlp.Linear( dim = 1, layer_name='y', irange=.0001 )

	layers = [l1, l2, output]
	nvis = x_train.shape[1]

	mdl = mlp.MLP( layers, nvis = nvis )	# input_space = input_space

	#lr = .001
	#epochs = 100
	
	decay = sgd.ExponentialDecay( decay_factor = decay_factor, min_lr = min_lr )

	trainer = sgd.SGD(
		learning_rate = learning_rate,
		batch_size=128,
		learning_rule=learning_rule.Momentum( momentum ),
		
		update_callbacks = [ decay ],

		# Remember, default dropout is .5
		cost = Dropout( input_include_probs = {'l1': l1_dropout},
				   input_scales={'l1': 1.}),

		#termination_criterion = EpochCounter(epochs),
		termination_criterion = MonitorBased(
			channel_name = "valid_objective",
			prop_decrease = 0.001,				# 0.1% of objective
			N = 10	
		),

		# valid_objective is MSE

		monitoring_dataset = { 'train': full, 'valid': valid }
	)

	watcher = best_params.MonitorBasedSaveBest( channel_name = 'valid_objective', save_path = output_model_file )
	
	experiment = Train( dataset = full, model = mdl, algorithm = trainer, extensions = [ watcher ] )
	experiment.main_loop()

	###

	error = get_error_from_model( output_model_file )
	print "*** error: {} ***".format( error )
	return error
	
	