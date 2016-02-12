from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, batch_norm, get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer
from lasagne.nonlinearities import leaky_rectify,sigmoid
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import theano
import theano.tensor as T


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model():

    dtensor4 = T.TensorType('float32', (False,)*4)
    input_var = dtensor4('inputs')
    dtensor1 = T.TensorType('float32', (False,)*1)
    target_var = dtensor1('targets')

    # input layer with unspecified batch size
    layer_input     = InputLayer(shape=(None, 30, 64, 64), input_var=input_var) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    layer_0         = DimshuffleLayer(layer_input, (0, 'x', 1, 2, 3))

    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_1         = batch_norm(Conv3DDNNLayer(incoming=layer_0, num_filters=64, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify))
    layer_2         = MaxPool3DDNNLayer(layer_1, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_3         = DropoutLayer(layer_2, p=0.25)

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_4         = batch_norm(Conv3DDNNLayer(incoming=layer_3, num_filters=128, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify))
    layer_5         = MaxPool3DDNNLayer(layer_4, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_6         = DropoutLayer(layer_5, p=0.25)

    # Recurrent layer
    layer_7         = DimshuffleLayer(layer_6, (0,2,1,3,4))
    layer_8         = LSTMLayer(layer_7, num_units=612, only_return_final=True)
    layer_9         = DropoutLayer(layer_8, p=0.25)

    # Output Layer
    layer_systole   = DenseLayer(layer_9, 600, nonlinearity=sigmoid)
    layer_diastole  = DenseLayer(layer_9, 600, nonlinearity=sigmoid)
    layer_output    = ConcatLayer([layer_systole, layer_diastole])

    # Loss
    prediction           = get_output(layer_output) 
    loss                 = squared_error(prediction, target_var)
    loss                 = loss.mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_output, trainable=True)
    updates              = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_output, deterministic=True)
    test_loss            = squared_error(test_prediction, target_var)
    test_loss            = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn             = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn               = theano.function([input_var, target_var], test_loss)

    # Compule a third function computing the prediction
    predict_fn           = theano.function([input_var], test_prediction)

    return [layer_output, train_fn, val_fn, predict_fn]










