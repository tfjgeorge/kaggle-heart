from sunnybrook.models.bdrx3DCNN import get_model as bottom_model
from theano import tensor
import numpy

def get_model(input_var, multiply_var):

	# on va arbitrairement choisir le 5eme qui correspond
	# à une coupe pas trop dégueu normalement

	test_prediction_bottom, prediction_bottom, params_bottom = \
		bottom_model(input_var[:,4], multiply_var)

	mult = theano.shared(numpy.array([1, 1]).astype('float32'))

	sums = prediction_bottom.sum(axis=(3, 4))
	maxs = sums.max(axis=(2))
	mins = sums.min(axis=(2))
	minmax = tensor.concatenate((maxs, mins))

	test_sums = test_prediction_bottom.sum(axis=(3, 4))
	test_maxs = test_sums.max(axis=(2))
	test_mins = test_sums.min(axis=(2))
	test_minmax = tensor.concatenate((test_maxs, test_mins))


	return test_minmax*mult*multiply_var, minmax*mult*multiply_var, params_bottom+[mult]