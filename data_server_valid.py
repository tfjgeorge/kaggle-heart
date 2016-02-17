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
	subset=slice(5002, 5292),
)

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 10)
)


downscaled_stream = RescaleMinDimension(
	stream, 90)

cropped_stream = RandomFixedSizeCrop(
    downscaled_stream, (80, 80), which_sources=('sax_features',))

float_stream = ScaleAndShift(cropped_stream, 1./1024, 0, which_sources=('sax_features',))
float32_stream = Cast(float_stream, 'floatX', which_sources=('sax_features',))

start_server(float32_stream, port=5558)