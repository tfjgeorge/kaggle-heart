from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from fuel.transformers import Flatten, ScaleAndShift#, Cast
#from fuel.transformers.image import Random2DRotation
from fuel.transformers.video import RescaleMinDimension
from custom_transformers_sunnybrook import RandomDownscale, RandomRotate, Cast, RandomLimit, Normalize, RandomFixedSizeCrop
import numpy
import math

train_set = H5PYDataset(
	'data_sunnybrook/sunnybrook_heart.hdf5',
	which_sets=('train',),
	subset=slice(40, 45),
	load_in_memory=True,
)

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 5)
)

resized_stream = RandomDownscale(stream, 70)
rotated_stream = RandomRotate(resized_stream, math.pi/10)

cropped_stream = RandomFixedSizeCrop(resized_stream, (64, 64))

limit_stream   = RandomLimit(cropped_stream, 12)
float_stream   = Normalize(limit_stream)
#float_stream   = ScaleAndShift(limit_stream, 1./1024, 0., which_sources=('image_features',))
float32_stream = Cast(float_stream, 'floatX')


#a = float32_stream.get_epoch_iterator()
#b = a.next()

start_server(float32_stream, port=5558, hwm=10)
