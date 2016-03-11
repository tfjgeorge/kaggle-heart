from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from fuel.transformers import Flatten, ScaleAndShift
from fuel.transformers.image import  Random2DRotation
from fuel.transformers.video import RescaleMinDimension
from custom_transformers_kaggle import RandomLimit, RandomFixedSizeCrop, Normalize, Cast, RandomRotate, ZeroPadding, OrderFeatures, ApplyMask
import math
import numpy


# number_train = 494 (counting valid set)

valid_set = H5PYDataset(
	'./data_kaggle/kaggle_heart.hdf5',
	which_sets=('train',),
	#subset=slice(451, 494), 
	subset=slice(451, 491), 
	load_in_memory=True
)

index_cases    = 0
index_position = 1
index_mult     = 2
index_sax      = 3
index_images   = 4
index_targets  = 5

stream = DataStream.default_stream(
    valid_set,
    iteration_scheme=ShuffledScheme(valid_set.num_examples, 10)
)

#downscaled_stream = RandomDownscale(stream, 70)
masked_stream     = ApplyMask(stream)
order_stream      = OrderFeatures(masked_stream)
cropped_stream    = RandomFixedSizeCrop(order_stream, (64,64))
float_stream      = Normalize(cropped_stream)
padded_stream     = ZeroPadding(float_stream)
casted_stream     = Cast(padded_stream, 'floatX')

start_server(casted_stream, port=5558, hwm=10)

