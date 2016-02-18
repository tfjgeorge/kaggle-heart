import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import dicom
from PIL import Image
import math
from fuel.datasets.hdf5 import H5PYDataset


def get_features(root_path):
   """Get path to all the frame in view SAX and contain complete features"""
   ret = []
   for root, _, files in os.walk(root_path):
       root=root.replace('\\','/')
       files=[s for s in files if ".dcm" in s]
       if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
           continue
       prefix = files[0].rsplit('-', 1)[0]
       fileset = set(files)
       expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
       if all(x in fileset for x in expected):
           ret.append([root + "/" + x for x in expected])
   # sort for reproduciblity
   return sorted(ret, key = lambda x: x[0])

def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = (float(arr[1]), float(arr[2]))
   return labelmap

def get_data(lst,preproc):
   data = []
   result = []
   for path in lst:
       f = dicom.read_file(path)
       img = f.pixel_array
       dst_path = path.rsplit(".", 1)[0] + ".64x64.jpg"
       # scipy.misc.imsave(dst_path, img)

       # resize to 70 px imgs
       original_height, original_width = img.shape[-2:]
       multiplier = max(70./ original_width, 70. / original_height)

       width = int(math.ceil(original_width * multiplier))
       height = int(math.ceil(original_height * multiplier))

       im = Image.fromarray(img.astype('int16'))
       im = numpy.array(im.resize((width, height))).astype('int16')

       result.append(dst_path)
       data.append(im)
   return [data,result]

labels = get_label_map("./data_kaggle/train.csv")
train_features = get_features('./data_kaggle/train')
submit_features = get_features('./data_kaggle/validate')

n_examples_train = 5293
n_examples_submit = 2128
n_total = n_examples_train + n_examples_submit

output_path = './data_kaggle/kaggle_heart.hdf5'
h5file = h5py.File(output_path, mode='w')
dtype = h5py.special_dtype(vlen=numpy.dtype('uint16'))

hdf_features = h5file.create_dataset('sax_features', (n_total,), dtype=dtype)
hdf_shapes = h5file.create_dataset('sax_features_shapes', (n_total, 3), dtype='int32')
hdf_cases = h5file.create_dataset('cases', (n_total, 1), dtype='int32')
hdf_labels = h5file.create_dataset('targets', (n_total, 2), dtype='float32')

# Attach shape annotations and scales
hdf_features.dims.create_scale(hdf_shapes, 'shapes')
hdf_features.dims[0].attach_scale(hdf_shapes)

hdf_shapes_labels = h5file.create_dataset('sax_features_labels', (3,), dtype='S7')
hdf_shapes_labels[...] = ['features'.encode('utf8'),
                          'height'.encode('utf8'),
                          'width'.encode('utf8')]
hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
hdf_features.dims[0].attach_scale(hdf_shapes_labels)

# Add axis annotations
hdf_features.dims[0].label = 'batch'
hdf_labels.dims[0].label = 'batch'
hdf_labels.dims[1].label = 'index'
hdf_cases.dims[0].label = 'batch'
hdf_cases.dims[1].label = 'index'

### loading train
i = 0

with progress_bar('train ', n_examples_train) as bar:
    for sequence in train_features:
        d = get_data(sequence, lambda x: x)
        images = numpy.array(d[0])

        hdf_features[i] = images.flatten().astype(numpy.dtype('uint16'))
        hdf_shapes[i] = images.shape

        path = d[1][1].split('/')
        hdf_labels[i] = numpy.array(labels[int(path[3])])
        hdf_cases[i] = int(path[3])

        i += 1
        bar.update(i)

### loading submit
with progress_bar('submit', n_examples_submit) as bar:
    for sequence in submit_features:
        d = get_data(sequence, lambda x: x)
        images = numpy.array(d[0])

        hdf_features[i] = images.flatten().astype(numpy.dtype('uint16'))
        hdf_shapes[i] = images.shape

        path = d[1][1].split('/')
        hdf_cases[i] = int(path[3])

        i += 1
        bar.update(i - n_examples_train)

# Add the labels
split_dict = {}
sources = ['sax_features', 'targets', 'cases']
for name, slice_ in zip(['train', 'submit'],
                        [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))
h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
