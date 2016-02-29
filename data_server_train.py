from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from fuel.transformers import Flatten, ScaleAndShift
from fuel.transformers.image import  Random2DRotation
from fuel.transformers.video import RescaleMinDimension
from custom_transformers_kaggle import RandomLimit, RandomFixedSizeCrop, Normalize, Cast, RandomRotate, ZeroPadding, RandomDownscale
import math
import math

# number_train = 494 (counting valid set)

train_set = H5PYDataset(
	'./data_kaggle/kaggle_heart.hdf5',
	which_sets=('train',),
	subset=slice(0, 450), #450
	load_in_memory=True
)

index_cases   = 0
index_mult    = 1
index_sax     = 2
index_images  = 3
index_targets = 4

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 10)
)

downscaled_stream = RandomDownscale(stream, 70)
cropped_stream    = RandomFixedSizeCrop(downscaled_stream, (64,64))
rotated_stream    = RandomRotate(cropped_stream, math.pi/10)
float_stream      = Normalize(cropped_stream)
padded_stream     = ZeroPadding(float_stream)
casted_stream     = Cast(padded_stream, 'floatX')

start_server(casted_stream, hwm=10)
