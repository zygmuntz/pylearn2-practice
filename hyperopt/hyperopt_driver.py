#!/usr/bin/env python

import numpy as np
import csv
from time import clock
from math import log

import hyperopt
from hyperopt import hp, fmin, tpe

import hyperopt_main

###

def run_test( x ):

	global writer
	global run_counter
	
	run_counter += 1
	print "run {}".format( run_counter )
	print "{}\n".format( x )

	error = hyperopt_main.main( x )	
	
	x = list( x )
	x.insert( 0, error )
	writer.writerow( x )
	
	return error

###

space = ( 
	hp.qloguniform( 'l1_dim', log( 10 ), log( 1000 ), 1 ), 
	hp.qloguniform( 'l2_dim', log( 10 ), log( 1000 ), 1 ),
	hp.loguniform( 'learning_rate', log( 1e-5 ), log( 1e-2 )),
	hp.uniform( 'momentum', 0.5, 0.99 ),
	hp.uniform( 'l1_dropout', 0.1, 0.9 ),
	hp.uniform( 'decay_factor', 1 + 1e-3, 1 + 1e-1 )
)

run_counter = 0
start_clock = clock()

output_file = 'results.csv'
writer = csv.writer( open( output_file, 'wb' ))

best = fmin( run_test, space, algo = tpe.suggest, max_evals = 50 )

print best
print run_test( hyperopt.space_eval( space, best ))

print "Seconds", clock() - start_clock
