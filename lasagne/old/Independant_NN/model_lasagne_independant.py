from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, FlattenLayer, DenseLayer, batch_norm, get_output, get_all_params, get_all_param_values
from lasagne.nonlinearities import leaky_rectify,softmax
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
import theano
import theano.tensor as T

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model():

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # input layer with unspecified batch size
    layer_0         = InputLayer(shape=(None, 30, 64, 64), input_var=input_var)

    # Z-score?

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_1         = batch_norm(Conv2DLayer(layer_0, 64, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_2         = batch_norm(Conv2DLayer(layer_1, 64, (3, 3), pad='valid', nonlinearity=leaky_rectify))
    layer_3         = MaxPool2DLayer(layer_2, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_4         = DropoutLayer(layer_3, p=0.25)

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_5         = batch_norm(Conv2DLayer(layer_4, 96, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_6         = batch_norm(Conv2DLayer(layer_5, 96, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_7         = MaxPool2DLayer(layer_6, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_8         = DropoutLayer(layer_7, p=0.25)

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_9         = batch_norm(Conv2DLayer(layer_8, 128, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_10        = batch_norm(Conv2DLayer(layer_9, 128, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_11        = MaxPool2DLayer(layer_10, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_12        = DropoutLayer(layer_11, p=0.25)

    # Last layers
    layer_13        = FlattenLayer(layer_12)
    layer_14        = DenseLayer(layer_13, 1024, nonlinearity=leaky_rectify)
    layer_15        = DropoutLayer(layer_14, p=0.5)
    layer_16        = DenseLayer(layer_15, 600, nonlinearity=softmax)

    # Loss
    prediction      = get_output(layer_16)
    loss            = squared_error(prediction, target_var)
    loss            = loss.mean() + regularize_layer_params(layer_14, l2)

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params          = get_all_params(layer_16, trainable=True)
    updates         = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction = get_output(layer_16, deterministic=True)
    test_loss       = squared_error(test_prediction, target_var)
    test_loss       = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn        = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn          = theano.function([input_var, target_var], test_loss)

    # Compule a third function computing the prediction
    predict_fn      = theano.function([input_var], test_prediction)

    return [layer_16, train_fn, val_fn, predict_fn]










