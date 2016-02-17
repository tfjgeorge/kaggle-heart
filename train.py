from fuel.streams import ServerDataStream
import theano
from theano import tensor
from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.algorithms import GradientDescent, Adam
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot
import datetime
import socket

train_stream = ServerDataStream(('cases','sax_features','targets'), False)

input_var = tensor.tensor4('sax_features')
target_var = tensor.matrix('targets')

from models.m2x2DCNN import get_model
prediction, loss, params = get_model(input_var, target_var)

loss.name = 'loss'

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
	# DataStreamMonitoring(variables=[loss, error], data_stream=valid_stream, prefix="valid"),
	Plot('%s %s @ %s' % ('test1', datetime.datetime.now(), socket.gethostname()), channels=[['loss', 'valid_loss_test'], ['valid_error']], after_epoch=True, server_url=host_plot),
	Printing(),
	# Checkpoint('train')
]

main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
