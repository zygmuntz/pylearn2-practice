"""
adult dataset wrapper for pylearn2
"""

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

class AdultDataset( DenseDesignMatrix ):

	def __init__(self, 
			path = 'train.csv',
			one_hot = False,
			with_labels = True,
			start = None,
			stop = None,
			preprocessor = None,
			fit_preprocessor = False,
			fit_test_preprocessor = False):
		"""
		which_set: A string specifying which portion of the dataset
			to load. Valid values are 'train' or 'public_test'
		base_path: The directory containing the .csv files from kaggle.com.
				This directory should be writable; if the .csv files haven't
				already been converted to npy, this class will convert them
				to save memory the next time they are loaded.
		fit_preprocessor: True if the preprocessor is allowed to fit the
				   data.
		fit_test_preprocessor: If we construct a test set based on this
					dataset, should it be allowed to fit the test set?
		"""

		# self._iter_targets = True	# whatever that means / won't work

		self.no_classes = 2

		# won't work TODO
		self.test_args = locals()
		self.test_args['which_set'] = 'test'
		self.test_args['fit_preprocessor'] = fit_test_preprocessor
		del self.test_args['start']
		del self.test_args['stop']
		del self.test_args['self']
		
		path = preprocess(path)
		X, y = self._load_data( path, with_labels )


		if start is not None:
			assert which_set != 'test'
			assert isinstance(start, int)
			assert isinstance(stop, int)
			assert start >= 0
			assert start < stop
			assert stop <= X.shape[0]
			X = X[start:stop, :]
			if y is not None:
				y = y[start:stop, :]


		super(AdultDataset, self).__init__(X=X, y=y)

		if preprocessor:
			preprocessor.apply(self, can_fit=fit_preprocessor)

	def _load_data(self, path, expect_labels):
	
		assert path.endswith('.csv')
	
		data = np.loadtxt( path, delimiter = ',', dtype = 'int' )
		
		if expect_labels:
			y = data[:,0]
			X = data[:,1:]

			# TODO: if one_hot
			# 10 is number of possible y values
			one_hot = np.zeros((y.shape[0], self.no_classes ),dtype='float32')
			for i in xrange( y.shape[0] ):
				label = y[i]
				if label == 1:
					one_hot[i,1] = 1.
				else:
					one_hot[i,0] = 1.
				
			y = one_hot
		else:
			X = data
			y = None

		return X, y
