from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, FlattenLayer, DenseLayer, batch_norm, get_output, get_all_params, ConcatLayer,RecurrentLayer
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
    layer_both_0         = InputLayer(shape=(None, 30, 64, 64), input_var=input_var)

    # Z-score?

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_both_1         = batch_norm(Conv2DLayer(layer_both_0, 64, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_2         = batch_norm(Conv2DLayer(layer_both_1, 64, (3, 3), pad='valid', nonlinearity=leaky_rectify))
    layer_both_3         = MaxPool2DLayer(layer_both_2, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_both_4         = DropoutLayer(layer_both_3, p=0.25)

    # Systole : Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_systole_0      = batch_norm(Conv2DLayer(layer_both_4, 96, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_systole_1      = batch_norm(Conv2DLayer(layer_systole_0, 96, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_systole_2      = MaxPool2DLayer(layer_systole_1, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_systole_3      = DropoutLayer(layer_systole_2, p=0.25)

    # Diastole : Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_diastole_0     = batch_norm(Conv2DLayer(layer_both_4, 96, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_diastole_1     = batch_norm(Conv2DLayer(layer_diastole_0, 96, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_diastole_2     = MaxPool2DLayer(layer_diastole_1, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_diastole_3     = DropoutLayer(layer_diastole_2, p=0.25)

    # Systole : Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_systole_4      = batch_norm(Conv2DLayer(layer_systole_3, 128, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_systole_5      = batch_norm(Conv2DLayer(layer_systole_4, 128, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_systole_6      = MaxPool2DLayer(layer_systole_5, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_systole_7      = DropoutLayer(layer_systole_6, p=0.25)

    # Diastole : Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_diastole_4     = batch_norm(Conv2DLayer(layer_diastole_3, 128, (3, 3), pad='same',nonlinearity=leaky_rectify))
    layer_diastole_5     = batch_norm(Conv2DLayer(layer_diastole_4, 128, (3, 3), pad='valid',nonlinearity=leaky_rectify))
    layer_diastole_6     = MaxPool2DLayer(layer_diastole_5, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_diastole_7     = DropoutLayer(layer_diastole_6, p=0.25)

    # Systole : Last layers
    layer_systole_8      = FlattenLayer(layer_systole_7)
    layer_systole_9      = DenseLayer(layer_systole_8, 1024, nonlinearity=leaky_rectify)
    layer_systole_10     = DropoutLayer(layer_systole_9, p=0.5)
    layer_systole_11     = DenseLayer(layer_systole_10, 600, nonlinearity=softmax)

    # Diastole : Last layers
    layer_diastole_8     = FlattenLayer(layer_diastole_7)
    layer_diastole_9     = DenseLayer(layer_diastole_8, 1024, nonlinearity=leaky_rectify)
    layer_diastole_10    = DropoutLayer(layer_diastole_9, p=0.5)
    layer_diastole_11    = DenseLayer(layer_diastole_10, 600, nonlinearity=softmax)

    # Add reccurrent layer and merge layer for output
    layer_recurrent      = RecurrentLayer(ConcatLayer([layer_systole_9, layer_diastole_9]), 512)
    layer_both_5         = ConcatLayer([layer_systole_11, layer_diastole_11])

    # Loss
    prediction           = get_output(layer_both_5) 
    loss                 = squared_error(prediction, target_var)
    loss                 = loss.mean() + regularize_layer_params(layer_systole_9, l2) + regularize_layer_params(layer_diastole_9, l2)

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_both_5, trainable=True)
    updates              = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_both_5, deterministic=True)
    test_loss            = squared_error(test_prediction, target_var)
    test_loss            = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn             = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn               = theano.function([input_var, target_var], test_loss)

    # Compule a third function computing the prediction
    predict_fn           = theano.function([input_var], test_prediction)

    return [layer_both_5, train_fn, val_fn, predict_fn]










