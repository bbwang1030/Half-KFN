# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from tensorflow import set_random_seed
seed = 2
np.random.seed(seed)
set_random_seed(seed)

import keras
import tempfile
import keras.models

from keras import backend as K
from utils.shift_detector import *
from utils.shift_locator import *
from utils.shift_applicator import *
from utils.data_utils import *
from utils.shared_utils import *
from resampling import *
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import utils.softmaxRegressionTrain as SRT
# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('axes', labelsize=22)
rc('xtick', labelsize=22)
rc('ytick', labelsize=22)
rc('legend', fontsize=13)

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#
#
# def loadModel(fileName):
# 	'''
#         loadModel()函数，根据文件名提取出模型theta，并返回一个n*1的矩阵
# 	输入:fileName文件名
# 	输出:一个(k*n)的矩阵,作为模型
# 	'''
# 	f=open(fileName)
# 	theta=[]
# 	for line in f.readlines():#遍历文件每一行(这里只有一行)
# 		tmpt=[]
# 		lines=line.strip().split("\t")#格式化后，用\t分割字符串
# 		for x in lines:
# 			tmpt.append(float(x))#将每一行的每一个字符串转为浮点数保存入列表中
# 		theta.append(tmpt)#将该行的值
# 	f.close()
# 	return np.mat(theta)
#
# def loadData(fileName,n):
#         '''
#         loadData()函数，通过文件名获取文件的数据,最后输出一个m*n的矩阵,作为样本
#         输入:fileName文件名
#         输出:一个(m*n)的矩阵，作为样本
#         '''
#         f=open(fileName)
#         x=[]
#         for line in f.readlines():
#                 tmpx=[]
#                 lines=line.strip().split("\t")
#                 if len(lines) != (n-1):#对于不合格的数据不要
#                         continue
#                 tmpx.append(1)#先加入一个常数项
#                 for i in lines:#加入后续的特征值
#                         tmpx.append(float(i))
#                 x.append(tmpx)
#         f.close()
#         return np.mat(x)
#
# def loadDatatrain(fileName):
#     '''
#     loadData()通过文件名导入训练数据
#     输出:返回一个x矩阵m*n的
#          返回一个y矩阵m*1的
#          返回一个y的标签数目len
#     '''
#     f=open(fileName)
#     x=[]
#     y=[]
#     for line in f.readlines():
#         tmpX=[]
#         tmpX.append(1)
#         lines=line.strip().split("\t")
#         for i in range(len(lines)-1):
#             tmpX.append(float(lines[i]))
#         y.append(int(float(lines[-1])))
#         x.append(tmpX)
#     f.close()
#     return np.mat(x),np.mat(y).T,len(set(y))
#
#
# def loadDatatrain_no1(fileName):
#     '''
#     loadData()通过文件名导入训练数据
#     输出:返回一个x矩阵m*n的
#          返回一个y矩阵m*1的
#          返回一个y的标签数目len
#     '''
#     f=open(fileName)
#     x=[]
#     y=[]
#     for line in f.readlines():
#         tmpX=[]
#         lines=line.strip().split("\t")
#         for i in range(len(lines)-1):
#             tmpX.append(float(lines[i]))
#         y.append(int(float(lines[-1])))
#         x.append(tmpX)
#     f.close()
#     return np.mat(x),np.mat(y).T,len(set(y))
#
# def loadDatatrain_have1(list):
#     '''
#     loadData()通过文件名导入训练数据
#     输出:返回一个x矩阵m*n的
#          返回一个y矩阵m*1的
#          返回一个y的标签数目len
#     '''
#     x=[]
#     y=[]
#     for line in list:
#         tmpX=[]
#         tmpX.append(1)
#         for i in line:
#             i = i[0].tolist()
#             tmpX.extend(i[0])
#         y.append(line[-1])
#         x.append(tmpX)
#     x = np.array(x)
#
#     return x,y
#
#

def predict(x,theta):
    '''
    predict()调用训练模型来预测结果
    输出:一个m*k的矩阵,表示的概率
    '''
    y=SRT.forecastFunction(theta,x)
    return y

def saveY(fileName,y):
    '''
    保存结果
    '''
    f=open(fileName,"w")
    m=np.shape(y)[0]
    k=np.shape(y)[1]
    for i in range(m):
        for j in range(k):
            f.write(str(y[i,j]))
            f.write("\t")
        f.write("\n")
    f.close()


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt='-o', label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill)


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
#

plt.rcParams['figure.figsize'] = (8, 10)
linestyles = [ '-.', '--', ':','-.', '--','-','-']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#FFB6C1', '#9acd32', '#eee8aa', '#8470ff', '#625b57', '#87cefa' ,'#f44336']
# colors = ['#87cefa', '#FFB6C1', '#9acd32', '#eee8aa', '#8470ff','#add8e6', '#625b57','#f44336', '#2196f3']
# -------------------------------------------------
# CONFIG
# -------------------------------------------------

make_keras_picklable()

datset = sys.argv[1]
test_type = sys.argv[3]

# Define results path and create directory.
path = './paper_results/'
path += test_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'
if not os.path.exists(path):
    os.makedirs(path)


# Define DR methods.
dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
if test_type == 'two_sample':
    dr_techniques = [DimensionalityReduction.BBSDs.value]
    # dr_techniques = [DimensionalityReduction.simulation.value]

# Define test types and general test sample sizes.
test_types = [td.value for td in TestDimensionality]
if test_type == 'two_sample':
    md_tests = [MultidimensionalTest.MMD.value,MultidimensionalTest.FR.value,MultidimensionalTest.Energy.value,MultidimensionalTest.KNN.value,MultidimensionalTest.SmoothKNN.value,MultidimensionalTest.halfKFN.value,MultidimensionalTest.KFN_ours.value]
    # md_tests = [MultidimensionalTest.halfKFN.value,MultidimensionalTest.KFN_ours.value]
    # md_tests = [MultidimensionalTest.MMD.value,MultidimensionalTest.Energy.value]
    od_tests = []
    # md_tests = [MultidimensionalTest.MMD.value,MultidimensionalTest.FR.value,MultidimensionalTest.Energy.value,MultidimensionalTest.KNN.value,MultidimensionalTest.SmoothKNN.value,MultidimensionalTest.halfKFN.value,MultidimensionalTest.KFN_ours.value]
    # samples = [10, 20, 50, 100, 200, 500, 1000]
    # samples = [1000]
    samples = [100,200,500,1000]


# Number of random runs to average results over.
# random_runs = 5
random_runs = 100

# Significance level.
sign_level = 0.05

# Define shift types.
if sys.argv[2] == 'medium_gn_shift':
    shifts = [
        # 'medium_gn_shift_0',
        'medium_gn_shift_0.01',
        'medium_gn_shift_0.05',
        # 'medium_gn_shift_0.1'
    ]
elif sys.argv[2] == 'large_gn_shift':
    shifts = [
        #'large_gn_shift_0',
        'large_gn_shift_0.01',
        'large_gn_shift_0.05',
        # 'large_gn_shift_0.1'
    ]
elif sys.argv[2] == 'adversarial_shift':
    shifts = [
        # 'adversarial_shift_0',
        'adversarial_shift_0.01',
        'adversarial_shift_0.05',
        'adversarial_shift_0.1'
    ]
elif sys.argv[2] == 'large_image_shift':
    shifts = [
        'large_img_shift_0',
        'large_img_shift_0.01',
        'large_img_shift_0.05',
        'large_img_shift_0.1'
    ]
elif sys.argv[2] == 'medium_img_shift+ko_shift':
    shifts = ['medium_img_shift_0.5+ko_shift_0.1',
              'medium_img_shift_0.5+ko_shift_0.5',
              'medium_img_shift_0.5+ko_shift_1.0']
    if test_type == 'univ':
        samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
elif sys.argv[2] == 'only_zero_shift+medium_img_shift':
    shifts = ['only_zero_shift+medium_img_shift_0.1',
              'only_zero_shift+medium_img_shift_0.5',
              'only_zero_shift+medium_img_shift_1.0']
    samples = [10, 20, 50, 100, 200, 500, 1000]
else:
    shifts = []

if datset == 'coil100' and test_type == 'univ':
    samples = [10, 20, 50, 100, 200, 500, 1000, 2400]

if datset == 'mnist_usps':
    samples = [10, 20, 50, 100, 200, 500, 1000]

if datset == 'simulation':
    samples = [100, 200, 500, 1000]
    # samples = [500]
    # 导入模型
    theta222 = loadModel("modelData")
    n222 = np.shape(theta222)[1]
    # 导入数据
    x1 = loadDatatrain("testData")[0]
    y = loadDatatrain("testData")[1]
    # 预测结果
    y1 = predict(x1, theta222)
    # 保存结果
    saveY("predicttest", y1)
    # 导入数据
    x2 = loadDatatrain("trainData")[0]
    print(x2)
    # 预测结果
    y2 = predict(x2, theta222)
    # 保存结果
    saveY("predicttrain", y2)
    x___ = loadDatatrain_no1("testData")[0]  # 去1

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

# Stores p-values for all experiments of a shift class.
# samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1)

red_dim = -1
red_models = [None] * len(DimensionalityReduction)

# Iterate over all shifts in a shift class.
for shift_idx, shift in enumerate(shifts):

    shift_path = path + shift + '/'
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores p-values for a single shift.
    rand_run_p_vals = np.ones((len(samples), len(md_tests), random_runs)) * (-1)
    rand_run_power = np.ones((len(samples), len(md_tests), random_runs)) * (-1)
    rand_run_time = np.ones((len(samples), len(md_tests), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    # Data preparation for resampling
    X_tr_red,y_tr_orig,X_te_red,X_all_red,y_all, nb_classes = data_preparation(datset,shift,shuffle=True)
    p2 = p2_(X_all_red,nb_classes)
    M = 10

    # Average over a few random runs to quantify robustness.
    for rand_run in range(random_runs):

        print("Random run %s" % rand_run)
        rand_run_path = shift_path + str(rand_run) + '/'
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)
        set_random_seed(rand_run)
        # Load data.
        (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
            import_dataset(datset, shuffle=True)
        if datset != 'simulation':
            X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            X_te_orig = normalize_datapoints(X_te_orig, 255.)
            X_val_orig = normalize_datapoints(X_val_orig, 255.)
        else:
            X_tr_orig = normalize_datapoints(X_tr_orig, 1.)
            X_te_orig = normalize_datapoints(X_te_orig, 1.)
            X_val_orig = normalize_datapoints(X_val_orig, 1.)

        # Apply shift.
        if datset != 'simulation':
            (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset)
        else:
            X_te_orig1 = X_te_orig
            y_te_orig1 = y_te_orig
            (X_te_1, y_te_1) = apply_shift(X_te_orig1, y_te_orig1, shift, orig_dims, datset)
            X_te_1 = loadDatatrain_have1(X_te_1)[0]  # 加1
            y_val_orig = y_val_orig.reshape(-1, 1)
           # X_val_orig, y_val_orig = random_shuffle(X_val_orig, y_val_orig)


        X_te_22, y_te_22 = random_shuffle(X_te_1, y_te_1)

        # Check detection performance for different numbers of samples from test.
        for si, sample in enumerate(samples):

            print("Sample %s" % sample)

            sample_path = rand_run_path + str(sample) + '/'
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            X_te_3 = X_te_22[:sample, :]
            y_te_3 = y_te_22[:sample]

            X_all_red, X_te_red = random_shuffle(X_all_red, X_te_red)
            X_tr_red_resample_list, X_te_red_resample_list = resampling(M, sample, X_te_red,X_all_red)

            X_val_3 = X_val_orig[:sample, :]
            y_val_3 = y_val_orig[:sample]

            X_tr_3 = np.copy(X_tr_orig)
            y_tr_3 = np.copy(y_tr_orig)
            # Detect shift.
            shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                           sample, datset)

            if datset == 'simulation':
                nb_classes = 3

            (od_decs, ind_od_decs, ind_od_p_vals), \
                (md_decs, ind_md_decs, ind_md_p_vals), \
                ind_md_delta, red_dim, red_models, val_acc, te_acc, ind_md_time = shift_detector.detect_data_shift(
                X_tr_3, X_tr_red_resample_list, y_tr_3,
                X_val_3, y_val_3,
                X_te_3, X_te_red_resample_list, y_te_3,
                orig_dims,
                nb_classes, p2,M)


            if test_type == 'two_sample':
                print("Shift decision: ", ind_md_decs.flatten())
                print("Shift p-vals: ", ind_md_p_vals.flatten())

                rand_run_p_vals[si,:,rand_run] = ind_md_p_vals.flatten()
                rand_run_time[si, :, rand_run] = ind_md_time.flatten()

        rand_run_power[(rand_run_p_vals < sign_level)&(rand_run_p_vals >= 0)] = 1
        rand_run_power[(rand_run_p_vals < 0) | (rand_run_p_vals >= sign_level)] = 0

        sum_power = np.sum(rand_run_power, axis=2) / random_runs
        sum_time= np.sum(rand_run_time, axis=2) / random_runs


    mean_p_vals = np.mean(rand_run_p_vals, axis=2)
    std_p_vals = np.std(rand_run_p_vals, axis=2)


    # plt.figure(figsize=(8, 10))
    for dr_idx, dr in enumerate(md_tests):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr], label="%s" % MultidimensionalTest(dr).name)
    plt.axhline(y=sign_level, color='k')
    plt.xlabel('Number of samples from test')
    plt.ylabel('$p$-value')
    plt.savefig("%s/dr_sample_comp_noleg.png" % shift_path, bbox_inches='tight')
    plt.legend()
    plt.savefig("%s/dr_sample_comp.png" % shift_path, bbox_inches='tight')
    plt.clf()

    for dr_idx, dr in enumerate(md_tests):
        plt.plot(np.array(samples), sum_power[:,dr_idx], format[dr], color=colors[dr], marker=markers[dr], linestyle = linestyles[dr], label="%s" % MultidimensionalTest_name(MultidimensionalTest(dr).name))
    plt.axhline(y=sign_level, color='k')
    plt.xlabel('Number of samples from test')
    plt.ylabel('Type I Error')
    # plt.ylabel('power')
    plt.xticks(ticks=np.array(samples))
    plt.yticks(ticks=np.linspace(0, 1, 11))
    plt.savefig("%s/power_sample_comp_noleg.png" % shift_path, bbox_inches='tight')
    plt.legend(loc=2,prop = {'size':16},ncol=2)
    plt.savefig("%s/power_sample_comp.png" % shift_path, bbox_inches='tight')
    plt.clf()


    for dr_idx, dr in enumerate(md_tests):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr])
        plt.xlabel('Number of samples from test')
        plt.ylabel('$p$-value')
        plt.axhline(y=sign_level, color='k', label='sign_level')
        plt.savefig("%s/%s_conf.png" % (shift_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        plt.clf()


    # plt.figure(figsize=(10, 6))
    for dr_idx, dr in enumerate(md_tests):
        # plt.figure(figsize=(8, 6))
        plt.plot(np.array(samples), sum_time[:,dr_idx], format[dr], color=colors[dr], marker=markers[dr], linestyle = linestyles[dr],
                 label="%s" % MultidimensionalTest_name(MultidimensionalTest(dr).name))
    plt.axhline(y=sign_level, color='k')
    plt.xlabel('Number of samples from test')
    plt.ylabel('CPU time')
    # plt.ylabel('power')
    plt.xticks(ticks=np.array(samples))
    plt.yticks(ticks=np.linspace(0, 10, 11))
    plt.savefig("%s/time_sample_comp_noleg.png" % shift_path, bbox_inches='tight')
    plt.legend(loc=2,prop = {'size':16},ncol=1)
    plt.savefig("%s/time_sample_comp.png" % shift_path, bbox_inches='tight')
    plt.clf()

    np.savetxt("%s/mean_p_vals.csv" % shift_path, mean_p_vals, delimiter=",")
    np.savetxt("%s/std_p_vals.csv" % shift_path, std_p_vals, delimiter=",")
    np.savetxt("%s/mean_power_vals.csv" % shift_path, sum_power, delimiter=",")
    # np.savetxt("%s/std_power_vals.csv" % shift_path, std_p_vals, delimiter=",")

    # for dr_idx, dr in enumerate(dr_techniques_plot):
    #     samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]
    #
    # np.save("%s/samples_shifts_rands_dr_tech.npy" % (path), samples_shifts_rands_dr_tech)
