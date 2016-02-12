from __future__ import print_function

import theano.tensor as T
from model import get_model
import sys
import numpy as np
from lasagne.layers import get_all_param_values
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, from_values_to_dirac


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train():
    """
    Training model.
    """

 	# Compile training and testing functions
    [model, train_fn, val_fn, predict_fn] = get_model()

    # Load training data
    print('Loading training data...')
    X, y = load_train_data()

    #print('Pre-processing images...')
    #X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)
    #y_train_systole                  = from_values_to_dirac(y_train[:, 0], n_range=600) # ADDED
    #y_train_diastole                 = from_values_to_dirac(y_train[:, 1], n_range=600) # ADDED
    #y_test_systole                   = from_values_to_dirac(y_test[:, 0], n_range=600)  # ADDED
    #y_test_diastole                  = from_values_to_dirac(y_test[:, 1], n_range=600)  # ADDED

    # concatenate systole and diastole outputs
    #y_train                          = np.concatenate((y_train_systole, y_train_diastole), axis=1)
    #y_test                           = np.concatenate((y_test_systole, y_test_diastole), axis=1)

    nb_epoch = 200
    batch_size = 32
    calc_crps = 0  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed) NOT IMPLEMENTED YET

    print('-'*50)
    print('Training...')
    print('-'*50)

    min_val_err  = sys.float_info.max
    patience     = 0
    for i in range(nb_epoch):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_epoch))
        print('-'*50)

        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        # In each epoch, we do a full pass over the training data:
        print('Fitting model...')
        train_err     = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train_aug, y_train, batch_size, shuffle=True):
            inputs, targets     = batch
            train_err          += train_fn(inputs, targets)
            train_batches      += 1

        # And a full pass over the validation data:
        val_err     = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
            inputs, targets     = batch
            val_err            += val_fn(inputs, targets)
            val_batches        += 1
        print('error on validation set: ' + str(val_err))
        print('patience variable is: ' + str(patience))
        print('\n')
        
        # assert(calc_crps == 0)
        # if calc_crps > 0 and i % calc_crps == 0:
        #     print('Evaluating CRPS...')
        #     train_pred        = predict_fn(X_train)
        #     val_pred          = predict_fn(X_test)

        #     pred_systole      = train_pred[:600]
        #     pred_diastole     = train_pred[600:]


        #     # CDF for train and test prediction
        #     pred_systole      = np.cumsum(pred_systole, axis=1)
        #     pred_diastole     = np.cumsum(pred_diastole, axis=1)
        #     val_pred_systole  = np.cumsum(val_pred_systole, axis=1)
        #     val_pred_diastole = np.cumsum(val_pred_diastole, axis=1)

        #     # evaluate CRPS on training data
        #     crps_train = crps(y_train, np.concatenate((pred_systole)))
        #     print('CRPS(train) = {0}'.format(crps_train))

        #     # evaluate CRPS on test data
        #     crps_test = crps(cdf_test, np.concatenate((val_pred_systole, val_pred_diastole)))
        #     print('CRPS(test) = {0}'.format(crps_test))

        print('Saving weights...')
        # save weights so they can be loaded later
        np.savez('weights.npz', *get_all_param_values(model))

        # for best (lowest) val losses, save weights
        if val_err < min_val_err:
            patience    = 0
            min_val_err = val_err
            np.savez('weights_best.npz', *get_all_param_values(model))
        else:
            patience   += 1

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss.txt', mode='a') as f:
            f.write(str(val_err))
            f.write('\n')
        
        if (patience == 5):
            break
train()
