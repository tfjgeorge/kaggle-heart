from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from fuel.transformers import Flatten, ScaleAndShift, Cast
from fuel.transformers.image import RandomFixedSizeCrop, Random2DRotation
from fuel.transformers.video import RescaleMinDimension
import numpy


train_set = H5PYDataset(
	'./data_kaggle/kaggle_heart.hdf5',
	which_sets=('train',),
	subset=slice(0, 4992),
	load_in_memory=True
)

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 32)
)


cropped_stream = RandomFixedSizeCrop(
    stream, (64, 64), which_sources=('sax_features',))

float_stream = ScaleAndShift(cropped_stream, 1./1024, 0, which_sources=('sax_features',))
float32_stream = Cast(float_stream, 'floatX', which_sources=('sax_features',))

start_server(float32_stream, hwm=10)
