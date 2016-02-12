from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, batch_norm, get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer
from lasagne.nonlinearities import softmax, leaky_rectify
from lasagne.objectives import squared_error
#from lasagne.updates import apply_nesterov_momentum,rmsprop
from lasagne.updates import adam
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import theano
import theano.tensor as T
from lasagne.init import Orthogonal


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model():

    dtensor4 = T.TensorType('float32', (False,)*4)
    input_var = dtensor4('inputs')
    dtensor2 = T.TensorType('float32', (False,)*2)
    target_var = dtensor2('targets')

    # input layer with unspecified batch size
    layer_input     = InputLayer(shape=(None, 30, 64, 64), input_var=input_var) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    layer_0         = DimshuffleLayer(layer_input, (0, 'x', 1, 2, 3))

    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_1         = batch_norm(Conv3DDNNLayer(incoming=layer_0, num_filters=64, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify, W=Orthogonal()))
    layer_2         = MaxPool3DDNNLayer(layer_1, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_3         = DropoutLayer(layer_2, p=0.25)

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_4         = batch_norm(Conv3DDNNLayer(incoming=layer_3, num_filters=128, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify, W=Orthogonal()))
    layer_5         = MaxPool3DDNNLayer(layer_4, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_6         = DropoutLayer(layer_5, p=0.25)
    
    # Recurrent layer
    layer_7         = DimshuffleLayer(layer_6, (0,2,1,3,4))
    layer_8         = LSTMLayer(layer_7, num_units=612, hid_init=Orthogonal(), only_return_final=False)

    # Output Layer
    layer_systole    = DenseLayer(layer_8, 600, nonlinearity=leaky_rectify, W=Orthogonal())
    layer_diastole   = DenseLayer(layer_8, 600, nonlinearity=leaky_rectify, W=Orthogonal())
    layer_systole_1  = DropoutLayer(layer_systole, p=0.3)
    layer_diastole_1 = DropoutLayer(layer_diastole, p=0.3)

    layer_systole_2   = DenseLayer(layer_systole_1, 1, nonlinearity=None, W=Orthogonal())
    layer_diastole_2  = DenseLayer(layer_diastole_1, 1, nonlinearity=None, W=Orthogonal())
    layer_output      = ConcatLayer([layer_systole_2, layer_diastole_2])

    # Loss
    prediction           = get_output(layer_output) 
    loss                 = squared_error(prediction, target_var)
    loss                 = loss.mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum Or Adam
    params               = get_all_params(layer_output, trainable=True)
    updates              = adam(loss, params)
    #updates_0            = rmsprop(loss, params)
    #updates              = apply_nesterov_momentum(updates_0, params)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_output, deterministic=True)
    test_loss            = squared_error(test_prediction, target_var)
    test_loss            = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn             = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy
    val_fn               = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)

    # Compule a third function computing the prediction
    predict_fn           = theano.function([input_var], test_prediction, allow_input_downcast=True)

    return [layer_output, train_fn, val_fn, predict_fn]










