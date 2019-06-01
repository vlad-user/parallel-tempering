import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from simulator import read_datasets
from simulator.simulator import Simulator
from simulator.summary_extractor import SummaryExtractor
from simulator.models.cifar10_models_v2 import lenet5_lr_with_dropout
from simulator import simulator_utils as s_utils


n_epochs = 400
learning_rate = 0.01
n_replicas = 8
noise_type = 'learning_rate'
dataset_name = 'cifar'
loss_func_name = 'cross_entropy'
train_data_size = 45000
model = lenet5_lr_with_const_dropout
model_name = 'lenet5'
description = 'reproduction of results'
beta_0 = .016
beta_n = .006
proba_coeff = 101000
batch_size = 128
burn_in_period_list = [10000, np.inf] # <-- simulate with and without swaps
swap_step = 600
mode = None 
test_step = 352 # <-- number of steps between computations of test error
timer = s_utils.Timer()

x_train, y_train, x_test, y_test, x_valid, y_valid = (
    read_datasets.get_emnist_letters())

timer = s_utils.Timer()
names = [] # <-- stores simulation names
for burn_in_period in burn_in_period_list:
  timer.start_timer()
  noise_list = np.linspace(beta_0, beta_n, n_replicas)
  noise_list = sorted(list(noise_list))

  name = s_utils.generate_experiment_name(model_name=model_name,
                                          dataset_name=dataset_name,
                                          separation_ratio=0,
                                          n_replicas=n_replicas,
                                          beta_0=beta_0,
                                          beta_n=beta_n,
                                          loss_func_name=loss_func_name,
                                          swap_step=swap_step,
                                          burn_in_period=burn_in_period,
                                          learning_rate=learning_rate,
                                          n_epochs=n_epochs,
                                          noise_type=noise_type,
                                          batch_size=batch_size,
                                          proba_coeff=proba_coeff,
                                          train_data_size=train_data_size,
                                          version='v5')
  names.append(name)
  scheduled_lr = {1: 0.1, 25000: 0.01} # annealing of learning rate

  sim = Simulator(model=model,
                  learning_rate=learning_rate,
                  noise_list=noise_list,
                  batch_size=batch_size,
                  n_epochs=n_epochs,
                  name=name,
                  burn_in_period=burn_in_period,
                  noise_type=noise_type,
                  scheduled_lr=scheduled_lr,
                  swap_step=swap_step,
                  separation_ratio=0,
                  n_simulations=1,
                  test_step=test_step,
                  loss_func_name=loss_func_name,
                  proba_coeff=proba_coeff,
                  mode=None)

  sim.train(train_data_size=train_data_size,
            train_data=x_train,
            train_labels=y_train,
            validation_data=x_valid,
            validation_labels=y_valid,
            test_data=x_test,
            test_labels=y_test)

  del sim
  gc.collect()
  print()
  log = 'Time took: {0:.3f}'.format(timer.elapsed_time())

SAMPLE_EVERY = 1

############# Lenet5 EMNIST + Learning Rate ###############
def get_min_err_and_rid(se):
  results = []
  for r in range(se.get_description()['n_replicas']):
      x, y = se.get_summary('test_error', replica_id=r)
      results.append(min(y))
  min_err = min(results)
  min_rid = np.argmin(results)
  return min_err, min_rid

def apply_filter(x, y, sigma=1):
    ynew = gaussian_filter1d(y, sigma=sigma)
    return x, ynew

EPOCH_MULT = np.ceil(train_data_size/batch_size)
xticks = [50000, 100000, 150000, 200000, 250000, 300000, 350000]
xlabels = ['50K', '100K', '150K', '200K', '250K', '300K', '350K']
LINEWIDTH = 5
TLINEWIDTH = 3
alpha = 0.35
sigma = 4
LEGEND_SIZE = 21
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

swap, noswap = names[0], names[1]
#noswap = 'lenet5_emnist_0_112320_8_0.88_crossentropy_2000_inf_0.01_400_128_dropoutgd_150_0.95_None_v4'
#swap = 'lenet5_emnist_0_112320_8_0.88_crossentropy_1000_25000_0.01_400_128_dropoutgd_500_0.95_None_v4'
se_swap = SummaryExtractor(swap)
se_noswap = SummaryExtractor(noswap)
noise_list = sorted(se_noswap.get_description()['noise_list'])

min_err_swap, swap_minrid = get_min_err_and_rid(se_swap)
min_err_noswap, noswap_minrid = get_min_err_and_rid(se_noswap)
noise_list = sorted(se_noswap.get_description()['noise_list'])
swapnoise_list = sorted(se_swap.get_description()['noise_list'])

xswap_test, yswap_test = se_swap.get_summary('test_error', replica_id=swap_minrid)
xnoswap_test, ynoswap_test = se_noswap.get_summary('test_error', replica_id=noswap_minrid)

xswap_train, yswap_train = se_swap.get_summary('train_error', replica_id=swap_minrid)
xnoswap_train, ynoswap_train = se_noswap.get_summary('train_error', replica_id=noswap_minrid)


fig, ax = plt.subplots(figsize=(12, 8))

yswap_test = 100*np.array(yswap_test)
ynoswap_test = 100*np.array(ynoswap_test)
yswap_train = 100*np.array(yswap_train)
ynoswap_train = 100*np.array(ynoswap_train)

batch_size = se_swap.get_description()['batch_size']
xswap_test = xswap_test*EPOCH_MULT
xnoswap_test = xnoswap_test*EPOCH_MULT
xswap_train = xswap_train*EPOCH_MULT
xnoswap_train = xnoswap_train*EPOCH_MULT

xswap_test = xswap_test
yswap_test = yswap_test
xnoswap_test = xnoswap_test
ynoswap_test = ynoswap_test
xswap_train = xswap_train
yswap_train = yswap_train
xnoswap_train = xnoswap_train
ynoswap_train = ynoswap_train




yswap_test_orig = yswap_test.copy()
yswap_train_orig = yswap_train.copy()
ynoswap_test_orig = ynoswap_test.copy()
ynoswap_train_orig = ynoswap_train.copy()

xswap_test, yswap_test = apply_filter(xswap_test, yswap_test, sigma=sigma)
xswap_train, yswap_train = apply_filter(xswap_train, yswap_train, sigma=sigma)
xnoswap_test, ynoswap_test = apply_filter(xnoswap_test, ynoswap_test, sigma=sigma)
xnoswap_train, ynoswap_train = apply_filter(xnoswap_train, ynoswap_train, sigma=sigma)

n_replicas = se_swap.get_description()['n_replicas']
label = '$\gamma\in {0} {1:.3f}, ..., {2:.3f} {3}^{4} $'.format(
    '\{', min(swapnoise_list), max(swapnoise_list), '\}', n_replicas)
ax.plot(xswap_test, yswap_test, label=label, color=colors[0], linewidth=LINEWIDTH)
ax.plot(xswap_test, yswap_test_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[0])


ax.plot(xnoswap_test, ynoswap_test, label='$\gamma^*={0:.3f}$)'.format(noise_list[noswap_minrid]),
        color=colors[1], linewidth=LINEWIDTH)
ax.plot(xnoswap_test, ynoswap_test_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[1])


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
FONTSIZE = 25


top_line_y = [32, 32]
ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
top_line_y = [30, 30]
ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
top_line_y = [28, 28]
ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))

plt.yticks([28, 30, 32])
xticks = [20000, 40000, 60000, 80000, 100000, 120000, 140000]
xlabels = ['20K', '40K', '60K', '80K', '100K', '120K', '140K']

plt.ylim((27.95, 32.05))
plt.xlim((0, 404*EPOCH_MULT))

plt.yticks(fontsize=23)
plt.xticks(xticks, xlabels, fontsize=23)
plt.xlabel('Mini-Batch Steps', fontsize=FONTSIZE)
plt.ylabel('Error (%)', fontsize=FONTSIZE)
plt.rcParams["legend.loc"] = 'lower left'
leg = plt.legend(fancybox=True, prop={'size': 19})
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(3)
ax.set_rasterized(True)

dirname = os.path.join(cwd, 'paper_results', 'plots')

if not os.path.exists(dirname):
  os.makedirs(dirname)

path = os.path.join(dirname, 'emnist-learning_rate.eps')

plt.savefig(path, bbox_inches='tight')
#plt.show()

