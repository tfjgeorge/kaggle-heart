from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from fuel.transformers import Flatten, ScaleAndShift
from fuel.transformers.image import  Random2DRotation
from fuel.transformers.video import RescaleMinDimension
from custom_transformers_kaggle import RandomLimit, RandomFixedSizeCrop, Normalize, Cast, RandomRotate, ZeroPadding
import math
import numpy


# number_train = 494 (counting valid set)

valid_set = H5PYDataset(
	'./data_kaggle/kaggle_heart.hdf5',
	which_sets=('train',),
	subset=slice(451, 494), 
	load_in_memory=True
)

index_cases   = 0
index_mult    = 1
index_sax     = 2
index_images  = 3
index_targets = 4

stream = DataStream.default_stream(
    valid_set,
    iteration_scheme=ShuffledScheme(valid_set.num_examples, 10)
)

cropped_stream = RandomFixedSizeCrop(stream, (64,64))
#rotated_stream = RandomRotate(cropped_stream, math.pi/10)
float_stream   = Normalize(cropped_stream)
padded_stream  = ZeroPadding(float_stream)
casted_stream  = Cast(padded_stream, 'floatX')

start_server(casted_stream, hwm=10)


start_server(float32_stream, port=5558, hwm=10)



##################################################### OLD #####################################################

# train_set = H5PYDataset(
# 	'./data_kaggle/kaggle_heart.hdf5',
# 	which_sets=('train',),
# 	subset=slice(5002, 5292),
# 	load_in_memory=True
# )

# stream = DataStream.default_stream(
#     train_set,
#     iteration_scheme=ShuffledScheme(train_set.num_examples, 10)
# )


# cropped_stream = RandomFixedSizeCrop(
#     stream, (64, 64), which_sources=('sax_features',))

# float_stream = ScaleAndShift(cropped_stream, 1./1024, 0, which_sources=('sax_features',))
# float32_stream = Cast(float_stream, 'floatX', which_sources=('sax_features',))