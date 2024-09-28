import numpy as np
from tensorflow import set_random_seed
seed = 2
np.random.seed(seed)
set_random_seed(seed)

from utils.shift_detector import *
from utils.shift_locator import *
from utils.shift_applicator import *
from utils.data_utils import *
from utils.shared_utils import *

def classify_data_all(X_tr, y_tr, X_val, y_val, X_te, y_te, X_all, y_all,datset, orig_dims):
    shift_reductor = ShiftReductor(X_tr, y_tr, X_val, y_val, DimensionalityReduction.BBSDs, orig_dims, datset, dr_amount=32)
    shift_reductor_model = shift_reductor.fit_reductor()
    X_te_red = shift_reductor.reduce(shift_reductor_model, X_te)
    X_tr_red = shift_reductor.reduce(shift_reductor_model, X_val)
    X_all_red = shift_reductor.reduce(shift_reductor_model, X_all)
    return X_tr_red,X_te_red,X_all_red

def classify_data_simulation(X_tr, y_tr, X_val, y_val, X_te, y_te,X_all, y_all,datset, orig_dims):
    shift_reductor = ShiftReductor(X_tr, y_tr, X_val, y_val, DimensionalityReduction.simulation, orig_dims, datset, dr_amount=32)
    shift_reductor_model = shift_reductor.fit_reductor()
    X_te_red = shift_reductor.reduce(shift_reductor_model, X_te)
    X_tr_red = shift_reductor.reduce(shift_reductor_model, X_val)
    X_all_red = shift_reductor.reduce(shift_reductor_model, X_all)
    return X_tr_red,X_te_red,X_all_red

#max min calculate P
def p2_(X_tr_red,nb_classes):
    X_tr_red = torch.tensor(X_tr_red)
    max_indices = np.argmax(X_tr_red, axis=1)
    # max_indices = X_tr_red.max(1).indices
    # print(X_te_red)
    p2 = 0
    for loc in range(nb_classes):
        rows_with_max_at_first_position = np.where(max_indices == loc)[0]
        X_tr_red_loc = X_tr_red[rows_with_max_at_first_position, :]
        print(X_tr_red_loc.shape)
        sample_one_dimension = torch.zeros([X_tr_red_loc.size(0)])
        for i in range(X_tr_red_loc.size(0)):
            # print(i, indices)
            # print( X_tr_red[i, indices])
            # sample_one_dimension[i] = X_tr_red[i, indices]
            sample_one_dimension[i] = X_tr_red_loc[i, loc]
        # print(sample_one_dimension)
        max = torch.max(sample_one_dimension)
        min = torch.min(sample_one_dimension)
        threshold = (max + min) / 2
        # 找到小于阈值的值的数量
        alpha_1 = torch.sum(sample_one_dimension < threshold).float()
        alpha_2 = torch.sum(sample_one_dimension >= threshold).float()
        p2_loc = (alpha_1 * (alpha_1 - 1) + alpha_2 * (alpha_2 - 1)) / ((alpha_1 + alpha_2) * (alpha_1 + alpha_2 - 1))
        q_ = (X_tr_red_loc.shape[0] / X_tr_red.shape[0]) * (X_tr_red_loc.shape[0] / X_tr_red.shape[0])
        print(X_tr_red_loc.shape[0] / X_tr_red.shape[0])
        p2 = p2 + p2_loc * q_
        print(max, min, alpha_1, alpha_2, X_tr_red.size(0), (alpha_1 * (alpha_1 - 1) + alpha_2 * (alpha_2 - 1)),
              (X_tr_red.size(0) * (X_tr_red.size(0) - 1)), p2)
    return p2

def data_preparation(datset,shift,shuffle=True):
    # Data preparation for bootsrap
    if datset != "simulation":
        (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
            import_dataset(datset, shuffle=shuffle)
        # print(X_tr_orig)
        X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
        X_te_orig = normalize_datapoints(X_te_orig, 255.)
        X_val_orig = normalize_datapoints(X_val_orig, 255.)
        x_all = np.concatenate((X_tr_orig, X_te_orig, X_val_orig), axis=0)
        y_all = np.concatenate((y_tr_orig, y_te_orig, y_val_orig), axis=0)
        x_all_te = np.concatenate((X_tr_orig, X_te_orig, X_val_orig), axis=0)

    else:
        nb_classes = 3
        X_val_orig = loadDatatrain("trainData")[0]
        X_val_orig_1 = loadDatatrain_no1("trainData")[0]
        y_val_orig = loadDatatrain("testData")[1]
        X_tr_orig = loadDatatrain("trainData")[0]
        X_tr_orig_1 = loadDatatrain_no1("trainData")[0]
        y_tr_orig = loadDatatrain("testData")[1]
        X_te_orig = loadDatatrain("testData")[0]  # have 1
        X_te_orig_1 = loadDatatrain_no1("testData")[0]  # no 1
        y_te_orig = loadDatatrain("testData")[1]
        orig_dims = 0
       # print("X_te_orig",X_te_orig.shape,X_tr_orig.shape,X__orig.shape)
        x_all = np.concatenate((X_tr_orig, X_te_orig, X_val_orig), axis=0)
        y_all = np.concatenate((y_tr_orig, y_te_orig, y_val_orig), axis=0)
        x_all_te = np.concatenate((X_tr_orig_1, X_te_orig_1, X_val_orig_1), axis=0)

    # Apply shift.
    if datset != 'simulation':
        (X_te_1, y_te_1) = apply_shift(x_all_te, y_all, shift, orig_dims, datset)
        # print(X_te_1)
    else:
        # X_te_1 = X_te_orig
        # y_te_1 = y_te_orig
        (X_te_1, y_te_1) = apply_shift(x_all_te, y_all, shift, orig_dims, datset)
        # print(X_te_1)
        X_te_1 = loadDatatrain_have1(X_te_1)[0]  # 加1
        y_val_orig = y_val_orig.reshape(-1, 1)
        X_val_orig, y_val_orig = random_shuffle(X_val_orig, y_val_orig)

    X_te_2, y_te_2 = random_shuffle(X_te_1, y_te_1)

    # estimate p2 of bootstrap
    if datset != 'simulation':
        X_tr_red, X_te_red,X_all_red = classify_data_all(X_tr_orig, y_tr_orig, X_val_orig, y_val_orig, X_te_2,
                                        y_te_2, x_all, y_all,datset, orig_dims)
    else:
        X_tr_red, X_te_red,X_all_red = classify_data_simulation(X_tr_orig, y_tr_orig, X_val_orig, y_val_orig, X_te_2,
                                            y_te_2,x_all, y_all,datset, orig_dims)
    X_tr_red = torch.tensor(X_tr_red)
    X_te_red = torch.tensor(X_te_red)
    X_all_red = torch.tensor(X_all_red)

    return X_tr_red,y_tr_orig,X_te_red,X_all_red,y_all, nb_classes

def resampling(M,sample,X_te_red,X_all_red):
    X_tr_red_resample_list = []
    X_te_red_resample_list = []
    for resample in range(M):
        np.random.seed(resample)
        set_random_seed(resample)
        X_all_red, X_te_red = random_shuffle(X_all_red, X_te_red)
        X_tr_red2 = X_all_red[:sample]
        X_te_red2 = X_te_red[sample:(2*sample)]
        X_tr_red_resample_list.append(X_tr_red2)
        X_te_red_resample_list.append(X_te_red2)
    return X_tr_red_resample_list, X_te_red_resample_list