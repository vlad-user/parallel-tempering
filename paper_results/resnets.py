import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
import gc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from simulator.simulator import Simulator
from simulator.models.cifar10_models import resnet
from simulator import simulator_utils as s_utils
from simulator import read_datasets

n_epochs = 182
learning_rate = 0.1
n_replicas = 8
noise_type = 'learning_rate_momentum'
model_name = 'resnet20'
dataset_name = 'cifar'
loss_func_name = 'cross_entropy'
train_data_size = 45000
n_simulations = 1 # run each simulation `n_simulations` times

description = 'reproduction of results'
beta_0 = 0.014
beta_n = 0.009
proba_coeff = 11
burn_in_period_list = [32000, np.inf]
batch_size = 128
swap_steps = [300, 3000]
resnet_size = 20
test_step = 352

x_train, y_train, x_test, y_test, x_valid, y_valid = (
        read_datasets._create_cifar_data_or_get_existing_resnet())

noise_list = np.linspace(beta_0, beta_n, n_replicas)
noise_list = sorted(list(noise_list))

scheduled_noise = {32000: noise_list,
                   1: [0.1 for _ in range(n_replicas)]}
resnet20_names = []
for burn_in_period, swap_step in zip(burn_in_period_list, swap_steps):
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
                                            mode=None)

    ensembles = resnet(tf.Graph(), n_replicas, resnet_size)
    resnet20_names.append(name)
    sim = Simulator(model=None,
                    learning_rate=learning_rate,
                    noise_list=noise_list,
                    noise_type=noise_type,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    name=name,
                    ensembles=ensembles,
                    burn_in_period=burn_in_period,
                    swap_step=swap_step,
                    separation_ratio=0,
                    n_simulations=n_simulations,
                    scheduled_noise=scheduled_noise,
                    test_step=test_step,
                    loss_func_name=loss_func_name,
                    proba_coeff=proba_coeff,
                    mode=None)
    os.system('clear')
    sim.train(train_data_size=train_data_size,
              train_data=x_train,
              train_labels=y_train,
              validation_data=x_valid,
              validation_labels=y_valid,
              test_data=x_test,
              test_labels=y_test)
    del sim
    del ensembles
    gc.collect()

n_epochs = 182
learning_rate = 0.1
n_replicas = 8
noise_type = 'learning_rate_momentum'
model_name = 'resnet44'
resnet_size = 44
dataset_name = 'cifar'
loss_func_name = 'cross_entropy'
train_data_size = 45000

description = 'reproduction of results'
beta_0 = 0.014
beta_n = 0.009
proba_coeff = 5
burn_in_period_list = [32000, np.inf]
batch_size = 128
swap_step = 300
test_step = 352

x_train, y_train, x_test, y_test, x_valid, y_valid = (
        read_datasets._create_cifar_data_or_get_existing_resnet())

noise_list = np.linspace(beta_0, beta_n, n_replicas)
noise_list = sorted(list(noise_list))

scheduled_noise = {32000: noise_list,
                   1: [0.1 for _ in range(n_replicas)]}
resnet44_names = []
for burn_in_period, swap_step in zip(burn_in_period_list, swap_steps):
    name = s_utils.generate_experiment_name(model=model_name,
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
                                            mode=None)

    ensembles = resnet(tf.Graph(), n_replicas, resnet_size)
    resnet44_names.append(name)
    sim = Simulator(model=None,
                    learning_rate=learning_rate,
                    noise_list=noise_list,
                    noise_type=noise_type,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    name=name,
                    ensembles=ensembles,
                    burn_in_period=burn_in_period,
                    swap_step=swap_step,
                    separation_ratio=0,
                    n_simulations=n_simulations,
                    scheduled_noise=scheduled_noise,
                    test_step=test_step,
                    loss_func_name=loss_func_name,
                    proba_coeff=proba_coeff,
                    mode=None)
    os.system('clear')
    sim.train(train_data_size=train_data_size,
              train_data=x_train,
              train_labels=y_train,
              validation_data=x_valid,
              validation_labels=y_valid,
              test_data=x_test,
              test_labels=y_test)
    del sim
    del ensembles
    gc.collect()

swap20, noswap20 = resnet20_names[0], renset20_names[1]
swap44, noswap44 = resnet44_names[0], resnet44_names[1]

se_swap20 = SummaryExtractor(swap20)
se_noswap20 = SummaryExtractor(noswap20)
se_swap44 = SummaryExtractor(swap44)
se_noswap44 = SummaryExtractor(noswap44)


noise_list20 = sorted(se_noswap20.get_description()['noise_list'])
noise_list44 = sorted(se_noswap44.get_description()['noise_list'])
swapnoise_list20 = sorted(se_swap20.get_description()['noise_list'])
swapnoise_list44 = sorted(se_noswap44.get_description()['noise_list'])


min_err_swap20, swap_minrid20 = get_min_err_and_rid(se_swap20)
min_err_noswap20, noswap_minrid20 = get_min_err_and_rid(se_noswap20)
min_err_swap44, swap_minrid44 = get_min_err_and_rid(se_swap44)
min_err_noswap, noswap_minrid44 = get_min_err_and_rid(se_noswap44)


xswap_test20, yswap_test20 = se_swap20.get_summary('test_error', replica_id=swap_minrid20, simulation_num=swap_sim_num20)
xnoswap_test20, ynoswap_test20 = se_noswap20.get_summary('test_error', replica_id=noswap_minrid20, simulation_num=noswap_sim_num20)

xswap_train20, yswap_train20 = se_swap20.get_summary('train_error', replica_id=swap_minrid20, simulation_num=swap_sim_num20)
xnoswap_train20, ynoswap_train20 = se_noswap20.get_summary('train_error', replica_id=noswap_minrid20, simulation_num=noswap_sim_num20)

xswap_test44, yswap_test44 = se_swap44.get_summary('test_error', replica_id=swap_minrid44)
xnoswap_test44, ynoswap_test44 = se_noswap44.get_summary('test_error', replica_id=noswap_minrid44)

xswap_train44, yswap_train44 = se_swap44.get_summary('train_error', replica_id=swap_minrid44)
xnoswap_train44, ynoswap_train44 = se_noswap44.get_summary('train_error', replica_id=noswap_minrid44)


yswap_test20 = 100*np.array(yswap_test20)
ynoswap_test20 = 100*np.array(ynoswap_test20)
yswap_train20 = 100*np.array(yswap_train20)
ynoswap_train20 = 100*np.array(ynoswap_train20)

xswap_test20 = xswap_test20*EPOCH_MULT
xnoswap_test20 = xnoswap_test20*EPOCH_MULT
xswap_train20 = xswap_train20*EPOCH_MULT
xnoswap_train20 = xnoswap_train20*EPOCH_MULT


yswap_test44 = 100*np.array(yswap_test44)
ynoswap_test44 = 100*np.array(ynoswap_test44)
yswap_train44 = 100*np.array(yswap_train44)
ynoswap_train44 = 100*np.array(ynoswap_train44)

xswap_test44 = xswap_test44*EPOCH_MULT
xnoswap_test44 = xnoswap_test44*EPOCH_MULT
xswap_train44 = xswap_train44*EPOCH_MULT
xnoswap_train44 = xnoswap_train44*EPOCH_MULT


times = 2
sigma = 3

yswap_test_orig20 = yswap_test20.copy()
yswap_train_orig20 = yswap_train20.copy()
ynoswap_test_orig20 = ynoswap_test20.copy()
ynoswap_train_orig20 = ynoswap_train20.copy()

yswap_test_orig44 = yswap_test44.copy()
yswap_train_orig44 = yswap_train44.copy()
ynoswap_test_orig44 = ynoswap_test44.copy()
ynoswap_train_orig44 = ynoswap_train44.copy()

xswap_test20, yswap_test20 = interpolate(xswap_test20, yswap_test20, sigma=sigma)
xswap_train20, yswap_train20 = interpolate(xswap_train20, yswap_train20, sigma=sigma)
xnoswap_test20, ynoswap_test20 = interpolate(xnoswap_test20, ynoswap_test20, sigma=sigma)
xnoswap_train20, ynoswap_train20 = interpolate(xnoswap_train20, ynoswap_train20, sigma=sigma)

xswap_test44, yswap_test44 = interpolate(xswap_test44, yswap_test44, sigma=sigma)
xswap_train44, yswap_train44 = interpolate(xswap_train44, yswap_train44, sigma=sigma)
xnoswap_test44, ynoswap_test44 = interpolate(xnoswap_test44, ynoswap_test44, sigma=sigma)
xnoswap_train44, ynoswap_train44 = interpolate(xnoswap_train44, ynoswap_train44, sigma=sigma)

alpha = 0.3
n_replicas = se_swap.get_description()['n_replicas']
label = '$ResNet 20\  \gamma\in {0} {1:.3f}, ..., {2:.3f} {3}^{4} $'.format(
    '\{', min(swapnoise_list20), max(swapnoise_list20), '\}', 8)
ax.plot(xswap_test20, yswap_test20, label=label, color=colors[0], linewidth=5)
ax.plot(xswap_test20, yswap_test_orig20, alpha=alpha, linewidth=5, color=colors[0])
ax.plot(xnoswap_test20, ynoswap_test20, label='ResNet20 $\gamma^*={0:.4f}$'.format(0.0114), color=colors[0], linewidth=4, linestyle='--')

n_replicas = se_swap.get_description()['n_replicas']
label = '$ResNet 44\  \gamma\in {0} {1:.3f}, ..., {2:.2f} {3}^{4} $'.format(
    '\{', min(swapnoise_list44), max(swapnoise_list44), '\}', n_replicas)
ax.plot(xswap_test44, yswap_test44, label=label, color=colors[2], linewidth=5)
ax.plot(xswap_test44, yswap_test_orig44, alpha=alpha, linewidth=5, color=colors[2])
ax.plot(xnoswap_test44, ynoswap_test44, label='Resnet44 $\gamma^*={0:.4f}$'.format(noise_list44[noswap_minrid44]), color=colors[2], linewidth=4, linestyle='--')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

FONTSIZE = 25


top_line_y = [14, 14]
ax.plot([0, 182*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
top_line_y = [10, 10]
ax.plot([0, 182*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
top_line_y = [7, 7]
ax.plot([0, 182*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))

plt.yticks([7, 10, 14])
xticks = [10000, 20000, 30000, 40000, 50000, 60000]
xlabels = ['10K', '20K', '30K', '40K', '50K', '60K']

plt.ylim((6.9, 14.05))
plt.xlim((0, 182*EPOCH_MULT))

plt.yticks(fontsize=23)
plt.xticks(xticks, xlabels, fontsize=23)
plt.xlabel('Mini-Batch Steps', fontsize=FONTSIZE)
plt.ylabel('Error (%)', fontsize=FONTSIZE)
plt.rcParams["legend.loc"] = 'upper right'
leg = plt.legend(fancybox=True, prop={'size': 20})
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(3)

ax.set_rasterized(True)
plt.savefig('_images/cifar-resnets.eps', bbox_inches='tight')
plt.show()
