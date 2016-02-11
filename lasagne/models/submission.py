from __future__ import print_function

import csv
import numpy as np

from model import get_model
from utils import real_to_cdf, preprocess
from lasagne.layers import set_all_param_values


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('data/X_validate.npy')
    ids = np.load('data/ids_validate.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, ids


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    print('Loading and compiling models...')
    [model, train_fn, val_fn, predict_fn] = get_model()

    print('Loading models weights...')

    with np.load('weights_best.hdf5.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    set_all_param_values(model, param_values)

    # load val losses to use as sigmas for CDF
    # with open('val_loss.txt', mode='r') as f:
    #     val_loss_systole = float(f.readline())
    #     val_loss_diastole = float(f.readline())

    print('Loading validation data...')
    X, ids = load_validation_data()

    print('Pre-processing images...')
    #X = preprocess(X)

    #batch_size = 32
    pred_systole     = np.zeros([X.shape[0], 600])
    pred_diastole    = np.zeros([X.shape[0], 600])
    print('Predicting on validation data...')
    nb_of_batches    = 10
    list_indexes     = np.linspace(0,X.shape[0],nb_of_batches + 1,dtype=np.int)

    for i in range(nb_of_batches):
        pred                                             = predict_fn(X[list_indexes[i]:list_indexes[i+1]])
        pred_systole[list_indexes[i]:list_indexes[i+1]]  = pred[:,:600]
        pred_diastole[list_indexes[i]:list_indexes[i+1]] = pred[:,600:]

    # CDF for train and test prediction
    cdf_pred_systole      = np.cumsum(pred_systole, axis=1)
    cdf_pred_diastole     = np.cumsum(pred_diastole, axis=1)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open('data/sample_submission_validate.csv'))
    f = open('submission.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

submission()
