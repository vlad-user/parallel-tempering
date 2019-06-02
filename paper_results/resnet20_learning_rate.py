import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
import gc

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

description = 'reproduction of results'
beta_0 = 0.014
beta_n = 0.009
proba_coeff = 11
burn_in_period_list = [32000, np.inf]
batch_size = 128
swap_step = 300
resnet_size = 20
test_step = 352

x_train, y_train, x_test, y_test, x_valid, y_valid = (
        read_datasets._create_cifar_data_or_get_existing_resnet())

noise_list = np.linspace(beta_0, beta_n, n_replicas)
noise_list = sorted(list(noise_list))

scheduled_noise = {32000: noise_list,
                   1: [0.1 for _ in range(n_replicas)]}

for burn_in_period in burn_in_period_list:
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
    