""" script that produces result for explaining neural data """


##
""" import modules """

import importlib
import os
import pickle
import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocess
import utils
import EBMs_numpy
import EBMs_numpy.rbm as rbm

mpl.style.use('ggplot')

path_log_dir = './model_log'
path_fig = './figures/thesis'

##

# load nmist data
(X_dtr, y_dtr), (X_dvl, y_dvl), (X_dts, y_dts) = utils.load_data()

m0, m1 = 784, 512
batchsize = 32

n_total = X_dtr.shape[0]

# unravel data for doing convolution:
X_dtr = utils.data_unravel(X_dtr)[:, :, :, None]
X_dvl = utils.data_unravel(X_dvl)[:, :, :, None]
X_dts = utils.data_unravel(X_dts)[:, :, :, None]

# one hot encoding of labels
ohe = preprocess.OneHotEncoder(sparse=False)
ohe.fit(y_dtr[:, None])
Y_dtr = ohe.transform(y_dtr[:, None])
Y_dvl = ohe.transform(y_dvl[:, None])
Y_dts = ohe.transform(y_dts[:, None])

# plot example data
importlib.reload(utils)
h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
for ax in h_ax.ravel():
    plt.axes(ax)
    utils.data_plot(X_dtr, y_dtr)


##
X_dvl_flp = X_dvl[:, :, ::-1]
X_dvl_rot = (np.rot90(X_dvl, k=1, axes=[1,2]))

X_tr = np.reshape(X_dtr, [X_dtr.shape[0], -1])
X_vl = np.reshape(X_dvl, [X_dvl.shape[0], -1])
X_vl_rot = np.reshape(X_dvl_rot, [X_dvl_rot.shape[0], -1])

X_tr = utils.data_binarize(X_tr, threshold=0.5, states='0,1')
X_vl = utils.data_binarize(X_vl, threshold=0.5, states='0,1')
X_vl_rot = utils.data_binarize(X_vl_rot, threshold=0.5, states='0,1')


utils.data_plot(X_vl, n=100, yn_random=False)
plt.suptitle('validation, original')
plt.gcf().set_size_inches(6,6)


utils.data_plot(X_vl_rot, n=100, yn_random=False)
plt.suptitle('validation, rotated')
plt.gcf().set_size_inches(6,6)


##

model = rbm.RestrictedBoltzmannMachine(m1=X_tr.shape[1], m2=128)

model.load_parameters(filepathname='./model_save/RBM_20180615_112719.h5')


##
""" plot trained filters """

def plot_W_from_pixel(model, i):
    H = int(np.sqrt(model.W.shape[0]))
    W = H

    toplot = model.W[:, i]
    utils.data_plot(toplot[None, :])
    climabs = np.abs(toplot).max()
    plt.clim(-climabs, climabs)
    plt.set_cmap('gray')

list_pixel_from = [0, 1, 2, 3]

# list_filters = [np.reshape(filter, [28, 28]) for filter in model.W.transpose()]
# plt.figure(figsize=(12, 6))
# utils.imshow_fast_subplot(list_filters, layout=(8, 16), cmap='gray')
# plt.axis('equal')

n_filters = model.W.shape[1]
h_fig, h_axes = plt.subplots(8, 16, sharex='all', sharey='all', figsize=(12, 6))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
h_axes = np.ravel(h_axes)
for i, h_ax in enumerate(h_axes):
    plt.axes(h_ax)
    plot_W_from_pixel(model, i)

plt.savefig(os.path.join(path_fig, 'learnt_weights.png'))
plt.savefig(os.path.join(path_fig, 'learnt_weights.pdf'))


##
""" energy distribution """

E_tr = model.cal_energy(X_tr[:2048])
E_vl = model.cal_energy(X_vl[:2048])
E_vl_rot = model.cal_energy(X_vl_rot[:2048])

plt.hist(E_tr, bins=20, density=True, alpha=0.3)
plt.hist(E_vl, bins=20, density=True, alpha=0.3)
plt.hist(E_vl_rot, bins=20, density=True, alpha=0.3)
plt.xlabel('energy')
plt.title('distribution of energy')
plt.legend(['train', 'val, original', 'val_rotated'])


##
""" generate samples """

i = 4
X = (np.random.random((1, X_tr.shape[1]))<0.1)*1.0
plt.figure(figsize=(8, 8))
for iter in range(25):
    plt.subplot(5, 5, iter+1)
    utils.data_plot(X)
    for temp in range(100):
        (X, Y), _ = model.inference(X, num_steps=10)


##
i = 0
X = X_vl[i:i+1, :]+0
utils.data_plot(X)

mask_visible = np.random.rand(1, X.shape[1]) < 0.5
X = np.where(mask_visible, X, np.random.rand(*X.shape) < 0.1)
plt.figure(figsize=(6, 6))
for iter in range(25):
    plt.subplot(5, 5, iter+1)
    utils.data_plot(X)
    for temp in range(10):
        (X, Y), _ = model.inference(X, mask_update=1-mask_visible, num_steps=2)


##
""" plot_dynamics """


inference_type = 'mask'
# inference_type = 'proportion'
proportion_original = 0.5

t_total = 49
n_samples = 10
num_examples = 10
# i_examples = np.random.choice(n_samples, size=num_examples)
i_examples = np.array([57, 83, 11, 0, 9, 19, 93, 75, 13, 99])
t_examples = range(0, 10, 1)


def compute_dyn(X, mask_visible, t_total=t_total):

    mask_update = 1 - mask_visible
    X_cur = np.where(mask_visible, X, np.random.random(X.shape) < np.mean(X))


    (_, Y_cur), _ = model.inference(X_cur, mask_update=mask_update, num_steps=1)
    Y_cur = Y_cur*0

    X_dyn = np.zeros(X_cur.shape + (t_total,))
    Y_dyn = np.zeros(Y_cur.shape + (t_total,))
    E_dyn = np.zeros((X_cur.shape[0], t_total))

    for t in range(t_total):
        X_dyn[:, :, t] = X_cur
        Y_dyn[:, :, t] = Y_cur
        E_dyn[:, t] = model.cal_energy(X_cur)
        if inference_type == 'mask':
            (X_cur, Y_cur), _ = model.inference(X_cur, mask_update=mask_update, num_steps=2)
        elif inference_type == 'proportion':
            (X_cur, Y_cur), _ = model.inference(X_cur, proportion_original=proportion_original, num_steps=2)
    return X_dyn, Y_dyn, E_dyn


def plot_example_dynamics(X_dyn, i_examples=None, t_examples=None, X_true=None):
    if i_examples is None:
        i_examples = np.random.choice(X_dyn.shape[0], size=10)
    if t_examples is None:
        t_examples = np.arange(0, X_dyn.shape[2], X_dyn.shape[2] // 10)

    X_plot = []

    for i, i_example in enumerate(i_examples):
        for j, t_example in enumerate(t_examples):
            X_plot.append(utils.data_unravel(X_dyn[i_example, :, t_example]))
    plt.figure(figsize=(8, 8), facecolor='gray')
    utils.imshow_fast_subplot(X_plot, cmap='gray')


def plot_dyn_stats(X_dyn, X_true, E_dyn, mask_visible, h_axes=None):
    if h_axes is None:
        h_fig, h_axes = plt.subplots(3, 1, figsize=(8, 8), facecolor='w', sharex='all')
        h_axes = np.ravel(h_axes)
    plt.axes(h_axes[0])
    plt.plot(np.mean(np.mean(np.abs(X_dyn[:, :, :] - X_true[:, :, None]), axis=1), axis=0)[1:])
    plt.ylabel("ave bias")
    plt.axes(h_axes[1])
    interval = 1
    plt.plot(
        np.mean(np.mean(np.abs(np.diff(X_dyn[:, :, :-interval] - X_dyn[:, :, interval:], axis=-1)), axis=1), axis=0)[
        1:])
    plt.ylabel("ave_flip")
    plt.axes(h_axes[2])
    plt.plot(np.mean(E_dyn, axis=0)[1:])
    plt.ylabel("system energy")
    plt.xlabel('interations')



n_samples = 100


mask_mode = 'random'
mask_prop_hidden = 0.3

X_in = X_vl[:n_samples, :] + 0


mask2D = np.random.random([28, 28]) > mask_prop_hidden

mask_visible = mask2D.ravel()[None, :]

X_dyn, Y_dyn, E_dyn = compute_dyn(X_in, mask_visible, t_total=t_total)

plot_example_dynamics(X_dyn, i_examples, t_examples)


h_fig, h_axes = plt.subplots(3, 1, figsize=(8, 8), facecolor='w', sharex='all')
h_axes = np.ravel(h_axes)
plot_dyn_stats(X_dyn, X_in, E_dyn, mask_visible, h_axes=h_axes)


##
""" get the resposne dynamics in different conditions (slow) """

n_samples = 2048

X_in = X_vl[:n_samples, :] + 0

list_cdtn = ['nov_10', 'nov_50', 'nov_70', 'fam_10', 'fam_50', 'fam_70']
# list_cdtn = ['nov_00', 'nov_30', 'nov_50', 'fam_00', 'fam_30', 'fam_50']

res_dyn_cdtn = dict()

for cdtn in list_cdtn:
    print(cdtn)

    if cdtn[:3] == 'fam':
        X_in = X_vl[:n_samples, :] + 0
    elif cdtn[:3] == 'nov':
        X_in = X_vl_rot[:n_samples, :] + 0
    else:
        raise Exception('cdtn not legal')
    label_in = y_dvl[:n_samples]

    mask_prop_hidden = int(cdtn[-2:])/100.0
    mask2D = np.random.random([28, 28]) > mask_prop_hidden
    mask_visible = mask2D.ravel()[None, :]
    X_dyn, Y_dyn, E_dyn = compute_dyn(X_in, mask_visible, t_total=t_total)

    res_dyn_cdtn[cdtn] = X_dyn, Y_dyn, E_dyn

if False:
    with open('./model_save/result_thesis_res_dyn_cdtn.pickle', 'wb') as f:
        pickle.dump(res_dyn_cdtn, f)


##
""" plot example dynamics """

for cdtn in list_cdtn:
    plot_example_dynamics(res_dyn_cdtn[cdtn][0], i_examples=i_examples, t_examples=t_examples)
    plt.savefig(os.path.join(path_fig, 'example_dynamics_{}.png'.format(cdtn)))


##
""" plot_mean_activity """
colors = np.vstack([utils.gen_distinct_colors(3, luminance=0.95, style='continuous', cm='rainbow'),
                    utils.gen_distinct_colors(3, luminance=0.65, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
focus_case = {'all': [1.0]*6, 'nov': [1.0]*3 + [0.0]*3, 'fam': [0.0]*3 + [1.0]*3}


# h_fig, h_axes = plt.subplots(3, 2, sharex='all', figsize=(6, 8))
# for i, cdtn in enumerate(list_cdtn):
#     plt.axes(h_axes[])
#     plt.plot(res_dyn_cdtn[cdtn][0].mean(axis=(0, 1)), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=3)
#
#     plt.axes(h_axes[1])
#     plt.plot(res_dyn_cdtn[cdtn][1].mean(axis=(0, 1)), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=3)
#
#     plt.axes(h_axes[2])
#     plt.plot(res_dyn_cdtn[cdtn][2].mean(axis=0), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=3)
#
# plt.axes(h_axes[0])
# plt.title('mean activity of visible layer')
# plt.axes(h_axes[1])
# plt.title('mean activity of hidden layer')
# plt.legend()
# plt.axes(h_axes[2])
# plt.title('system energy')

# plt.xlim(0, 30)
# plt.legend()
# plt.savefig(os.path.join(path_fig, 'activity_dynamics.png'))
# plt.savefig(os.path.join(path_fig, 'activity_dynamics.pdf'))


h_fig, h_axes = plt.subplots(2, 3, sharex='all', sharey='row', figsize=(12, 8))

for i_focus, focus in enumerate(['all', 'nov', 'fam']):
    plt.axes(h_axes[0, i_focus])
    plt.title('ave_activity_{}'.format(focus))
    for i_cdtn, cdtn in enumerate(list_cdtn):
        plt.plot(res_dyn_cdtn[cdtn][1].mean(axis=(0, 1)), label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

    plt.axes(h_axes[1, i_focus])
    plt.title('energy_{}'.format(focus))
    for i_cdtn, cdtn in enumerate(list_cdtn):
        plt.plot(res_dyn_cdtn[cdtn][2].mean(axis=0), label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

plt.axes(h_axes[0, 0])
plt.ylim(0.25, 0.5)
plt.ylabel('ave_activity')
plt.legend()

plt.axes(h_axes[1, 0])
plt.ylabel('energy')
plt.legend()

plt.xlim(0, 30)

plt.savefig(os.path.join(path_fig, 'activity_dynamics.png'))
plt.savefig(os.path.join(path_fig, 'activity_dynamics.pdf'))


##
""" plot tuning curve """

X_dyn, Y_dyn, E_dyn = res_dyn_cdtn['fam_10']
Y_mean_activity = np.mean(Y_dyn, axis=-1)
indx_sort_fam = np.argsort(Y_mean_activity, axis=0)[::-1]
smooth_range =10

X_dyn, Y_dyn, E_dyn = res_dyn_cdtn['nov_10']
Y_mean_activity = np.mean(Y_dyn, axis=-1)
indx_sort_nov = np.argsort(Y_mean_activity, axis=0)[::-1]

def sort_activity(activity, indx_sort):
    activity_sort = activity*0
    for i in range(activity.shape[1]):
        activity_sort[:, i] = activity[indx_sort[:, i], i]
    return activity_sort

h_fig, h_axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(8, 4), squeeze=False)
h_axes = h_axes.ravel()
for i, cdtn in enumerate(list_cdtn):
    if cdtn[:3] == 'fam':
        indx_sort = indx_sort_fam
    elif cdtn[:3] == 'nov':
        indx_sort = indx_sort_nov
    else:
        raise Exception('cdtn is illegal')

    X_dyn, Y_dyn, E_dyn = res_dyn_cdtn[cdtn]

    Y_mean_activity = np.mean(Y_dyn[:, :, :5], axis=-1)
    tuning_curve = np.mean(sort_activity(Y_mean_activity, indx_sort), axis=1)
    tuning_curve_smooth = ndimage.gaussian_filter1d(tuning_curve, sigma=smooth_range, mode='reflect')

    plt.axes(h_axes[0])
    plt.plot(tuning_curve_smooth, label=cdtn,
             linestyle=linestyles[i], color=colors[i], linewidth=2)

    Y_mean_activity = np.mean(Y_dyn[:, :, :], axis=-1)
    tuning_curve = np.mean(sort_activity(Y_mean_activity, indx_sort), axis=1)
    tuning_curve_smooth = ndimage.gaussian_filter1d(tuning_curve, sigma=smooth_range, mode='reflect')
    plt.axes(h_axes[1])
    plt.plot(tuning_curve_smooth, label=cdtn,
             linestyle=linestyles[i], color=colors[i], linewidth=2)


plt.axes(h_axes[0])
plt.title('tuning_curve, early')
plt.axes(h_axes[1])
plt.title('tuning_curve, late')

plt.legend()
plt.savefig(os.path.join(path_fig, 'tuning_curve.png'))
plt.savefig(os.path.join(path_fig, 'tuning_curve.pdf'))



##
""" population decoding (slow) """
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
t_size_window = 5
ts_windowed =  np.arange(t_total)[t_size_window//2::t_size_window]

def decoding_score(Y_dyn, y, return_p=False):

    Y_dyn_window_count = ndimage.convolve1d(Y_dyn, np.ones(t_size_window), mode='reflect', axis=2)
    Y_dyn_window_count = Y_dyn_window_count[:, :, t_size_window//2::t_size_window]


    N_ts = Y_dyn_window_count.shape[2]
    clf = linear_model.LogisticRegression(solver='lbfgs', warm_start=True, multi_class='multinomial',
                                          fit_intercept=False, n_jobs=4)
    clf_score = np.zeros(N_ts)

    if return_p:
        def get_score_p(clf, X, Y):
            dict_label2indx = dict(zip(clf.classes_, np.arange(len(clf.classes_))))
            Y_indx = np.array([dict_label2indx[y] for y in Y])
            proba_all = clf.predict_proba(X)
            proba_target = proba_all[np.arange(len(Y)), Y_indx]
            return np.mean(proba_target)

        object_cv = model_selection.StratifiedKFold(n_splits=4)
        score_cv = []
        for t in range(N_ts):
            X_cur = Y_dyn_window_count[:, :, t]
            score_all_fold = []
            for indx_train, indx_test in object_cv.split(X_cur, y):
                clf.fit(X_cur[indx_train, :], y[indx_train])
                score_all_fold.append(get_score_p(clf, X_cur[indx_test], y[indx_test]))
            clf_score[t] = np.mean(score_all_fold)

    else:
        for t in range(N_ts):
            cfl_scores = model_selection.cross_val_score(clf, Y_dyn_window_count[:, :, t], y, cv=4)
            clf_score[t] = np.mean(cfl_scores)
            # clf_score_std[t] = np.std(cfl_scores)

    return clf_score

res_decoding_cdtn = dict()
for cdtn in list_cdtn:
    print(cdtn)
    decoding_cur = decoding_score(res_dyn_cdtn[cdtn][1], label_in, return_p=True)
    res_decoding_cdtn[cdtn] = decoding_cur


if False:
    with open('./model_save/result_thesis_res_decoding_cdtn.pickle', 'wb') as f:
        pickle.dump(res_decoding_cdtn, f)


##
# h_fig, h_axes = plt.subplots(1, 1, sharex='all', figsize=(5, 4), squeeze=False)
# h_axes = h_axes.ravel()
# for i, cdtn in enumerate(list_cdtn):
#     plt.axes(h_axes[0])
#     plt.plot(ts_windowed, res_decoding_cdtn[cdtn], label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=2)
#
# plt.axes(h_axes[0])
# plt.title('decoding score')

h_fig, h_axes = plt.subplots(1, 3, sharex='all', sharey='row', figsize=(12, 4))
for i_focus, focus in enumerate(['all', 'nov', 'fam']):
    plt.axes(h_axes[i_focus])
    plt.title('decoding_{}'.format(focus))
    for i_cdtn, cdtn in enumerate(list_cdtn):
        plt.plot(ts_windowed, res_decoding_cdtn[cdtn], label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

plt.axes(h_axes[0])

plt.ylabel('decoding_performance')

plt.legend()


plt.xlim(0, 40)
plt.legend()
plt.savefig(os.path.join(path_fig, 'decoding_dynamics.png'))
plt.savefig(os.path.join(path_fig, 'decoding_dynamics.pdf'))



##
""" cross_trial_variability """

n_samples = 128
n_repeat =16

X_in = X_vl[:n_samples, :] + 0

list_cdtn = ['nov_10', 'nov_50', 'nov_70', 'fam_10', 'fam_50', 'fam_70']

res_dyn_cdtn_rep = dict()

for cdtn in list_cdtn:
    print(cdtn)

    if cdtn[:3] == 'fam':
        X_in = X_vl[:n_samples, :] + 0
    elif cdtn[:3] == 'nov':
        X_in = X_vl_rot[:n_samples, :] + 0
    else:
        raise Exception('cdtn not legal')

    X_dyn_all = []
    Y_dyn_all = []
    E_dyn_all = []

    mask_prop_hidden = int(cdtn[-2:]) / 100.0
    mask2D = np.random.random([28, 28]) > mask_prop_hidden
    mask_visible = mask2D.ravel()[None, :]

    for i_repeat in range(n_repeat):
        X_dyn, Y_dyn, E_dyn = compute_dyn(X_in, mask_visible, t_total=t_total)

        X_dyn_all.append(X_dyn)
        Y_dyn_all.append(Y_dyn)
        E_dyn_all.append(E_dyn)

    X_dyn_repeat = np.stack(X_dyn_all, axis=-1)
    Y_dyn_repeat = np.stack(Y_dyn_all, axis=-1)
    E_dyn_repeat = np.stack(E_dyn_all, axis=-1)

    res_dyn_cdtn_rep[cdtn] = X_dyn_repeat, Y_dyn_repeat, E_dyn_repeat


if False:
    with open('./model_save/result_thesis_res_decoding_cdtn_rep.pickle', 'wb') as f:
        pickle.dump(res_dyn_cdtn_rep, f)

##
""" fano over time """
t_size_window = 3



h_fig, h_axes = plt.subplots(3, 3, sharex='all', sharey='row', figsize=(12, 12))

for i_focus, focus in enumerate(['all', 'nov', 'fam']):
    plt.axes(h_axes[0, i_focus])

    for i_cdtn, cdtn in enumerate(list_cdtn):
        X_dyn, Y_dyn, E_dyn = res_dyn_cdtn_rep[cdtn]

        Y_dyn_window_count = ndimage.convolve1d(Y_dyn, np.ones(t_size_window), mode='reflect', axis=2)
        var_Y = np.var(Y_dyn_window_count, axis=-1)
        mean_Y = np.mean(Y_dyn_window_count, axis=-1)
        fano_Y = var_Y / (mean_Y + 10e-3)

        plt.axes(h_axes[0, i_focus])
        plt.plot(mean_Y.mean(axis=(0, 1)), label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

        plt.axes(h_axes[1, i_focus])
        plt.plot(var_Y.mean(axis=(0, 1)), label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

        plt.axes(h_axes[2, i_focus])
        plt.plot(fano_Y.mean(axis=(0, 1)), label=cdtn,
                 linestyle=linestyles[i_cdtn], color=colors[i_cdtn], linewidth=3, alpha=focus_case[focus][i_cdtn])

plt.axes(h_axes[0, 0])
plt.title('mean')
plt.ylabel('mean')
plt.legend()

plt.axes(h_axes[1, 0])
plt.title('var')
plt.ylabel('var')

plt.legend()
plt.axes(h_axes[2, 0])
plt.title('Fano factor')
plt.ylabel('Fano_factor')

plt.suptitle('cross-trial variability')

plt.xlim(0, 40)

plt.savefig(os.path.join(path_fig, 'variability.png'))
plt.savefig(os.path.join(path_fig, 'variability.pdf'))




# h_fig, h_axes = plt.subplots(3, 1, sharex='all', figsize=(8, 10), squeeze=False)
# h_axes = h_axes.ravel()
# for i, cdtn in enumerate(list_cdtn):
#     X_dyn, Y_dyn, E_dyn = res_dyn_cdtn_rep[cdtn]
#
#     Y_dyn_window_count = ndimage.convolve1d(Y_dyn, np.ones(t_size_window), mode='reflect', axis=2)
#     var_Y = np.var(Y_dyn_window_count, axis=-1)
#     mean_Y = np.mean(Y_dyn_window_count, axis=-1)
#     fano_Y = var_Y / (mean_Y + 10e-3)
#
#     plt.axes(h_axes[0])
#     plt.plot(mean_Y.mean(axis=(0, 1)), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=2)
#
#     plt.axes(h_axes[1])
#     plt.plot(var_Y.mean(axis=(0, 1)), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=2)
#
#     plt.axes(h_axes[2])
#     plt.plot(fano_Y.mean(axis=(0, 1)), label=cdtn,
#              linestyle=linestyles[i], color=colors[i], linewidth=2)
#
# plt.axes(h_axes[0])
# plt.title('mean')
# plt.axes(h_axes[1])
# plt.title('var')
# plt.axes(h_axes[2])
# plt.title('Fano factor')
#
# plt.suptitle('cross-trial variability')
#
# plt.xlim(0, 40)
# plt.legend()
# plt.savefig(os.path.join(path_fig, 'variability.png'))
# plt.savefig(os.path.join(path_fig, 'variability.pdf'))



