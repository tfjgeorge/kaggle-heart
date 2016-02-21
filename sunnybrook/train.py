from fuel.streams import ServerDataStream
import theano
from theano import tensor
from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.algorithms import GradientDescent, Adam
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot
import datetime
import time
import sys
import socket
import theano.tensor as T



def run(get_model, model_name):
	train_stream = ServerDataStream(('cases', 'image_features', 'image_targets', 'multiplier'), False, hwm=10)
	valid_stream = ServerDataStream(('cases', 'image_features', 'image_targets', 'multiplier'), False, hwm=10, port=5558)

	input_var  = tensor.tensor4('image_features')
	target_var = tensor.tensor4('image_targets')
	multiply_var = tensor.matrix('multiplier')
	multiply_var = T.addbroadcast(multiply_var, 1)

	prediction, crps, loss, params = get_model(input_var, target_var, multiply_var)

	loss.name = 'loss'
	crps.name = 'crps'

	algorithm = GradientDescent(
		cost=loss,
		parameters=params,
		step_rule=Adam(),
		on_unused_sources='ignore'
	)

	host_plot = 'http://localhost:5006'

	extensions = [
		Timing(),
		TrainingDataMonitoring([loss], after_epoch=True),
		DataStreamMonitoring(variables=[crps, loss], data_stream=valid_stream, prefix="valid"),
		Plot('%s %s %s' % (model_name, datetime.date.today(), time.strftime('%H:%M')), channels=[['loss','valid_loss'], ['valid_crps']], after_epoch=True, server_url=host_plot),
		Printing(),
		Checkpoint('train'),
		FinishAfter(after_n_epochs=20)
	]

	main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
	                     extensions=extensions)
	main_loop.run()



if __name__ == "__main__":

	if len(sys.argv) != 2:
		print('Usage: python train.py path/to/model.py')
		exit()

	# prepare path for import
	path = sys.argv[-1]
	if path[-3:] == '.py':
		path = path[:-3]
	path = path.replace('/','.')
	
	# import right model
	get_model = __import__(path, globals(), locals(), ['get_model']).get_model

	# run the training
	run(get_model, path.split('.')[-1])
