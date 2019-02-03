# Parallel Tempering

Parallel Tempering is a Monte-Carlo simulation method of sampling a physical system which runs multiple copies of that system, randomly initialized and at different temperatures. Then, based on some probability criterion one exchanges configurations (temperatures) of two systems with adjacent temperatures. In statistical physics, this method allows improved learning of a phase/configuration space because systems do not get trapped in local minimas and continue to sample additional volumes of space resulting in less biased empirical probability distribution. Here we apply this method for finding a global minimizer for a non-convex functions in problems arising in deep learning.

### Requirements

* [Python >= 3.5](https://www.python.org/)
* [TensorFlow 1.9.0 or 1.12.0](https://www.tensorflow.org/).
* [scikit-learn](https://scikit-learn.org/stable/) - Used for shuffling datasets.
* [NumPy](http://www.numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) - (Optional) For plotting simulated results.
* [PyLatex](https://jeltef.github.io/PyLaTeX/current/) - (Optional) For generating pdf files with simulated results.

## Examples

### 1. Simulate mnist dataset with multilayer perceptron using Langevin dynamics optimizer
```python
from simulator import read_datasets
from simulator.simulator import Simulator
from simulator.summary_extractor import SummaryExtractor
from simulator.models.mnist_models import nn_mnist_model

train_data, train_labels, test_data, test_labels, valid_data, valid_labels = (
    read_datasets.get_mnist_data())

# set hyper-parameters
n_replicas = 8
test_step = 200
learning_rate = 0.01
batch_size = 50
separation_ratio = 2.7
burn_in_period = 4000
proba_coeff = 0.0001
swap_step = 400
beta_0 = 900
name = 'test_simulation'
model = nn_mnist_model_small
noise_list = [beta_0*separation_ratio**i for i in range(n_replicas)]
noise_type = 'langevin'
n_epochs = 1000

# create and run simulation
sim = Simulator(model=model,
                learning_rate=learning_rate,
                noise_list=noise_list,
                noise_type=noise_type,
                batch_size=batch_size,
                n_epochs=n_epochs,
                name=name,
                burn_in_period=burn_in_period,
                test_step=test_step,
                swap_step=swap_step,
                loss_func_name='cross_entropy',
                proba_coeff=proba_coeff)

sim.train(train_data_size=7000,
          train_data=train_data,
          train_labels=train_labels,
          test_data=test_data,
          test_labels=test_labels,
          validation_data=valid_data,
          validation_labels=valid_labels)

# print report
se = SummaryExtractor(name)
se.show_report()
```

### 2. Simulate the same setup multiple times

```python

import os

import tensorflow as tf
import numpy as np

from simulator import read_datasets
from simulator.simulator import Simulator
from simulator.summary_extractor import SummaryExtractor
from simulator.models.mnist_models import nn_mnist_model_small
from simulator import simulator_utils as s_utils

# static hyper-parameters
n_epochs = 5000
learning_rate  = 0.01
n_replicas = 8
noise_type = 'langevin'
func_name = 'nn'
dataset_name = 'mnist'
loss_func_name = 'cross_entropy'
train_data_size = 7000
model = nn_mnist_model_small
n_simulations = 5
description = 'Testing different hyper-params.'

# varying hyper-parameters
sep_ratio_list = [1.5, 1.8, 2.1, 2.5, 2.8]
beta_0_list = [200, 1000]
proba_coeff_list = [0.005, 0.01]
swap_step_list = [100, 200]
burn_in_period_list = [1000]
batch_size_list = [50, 500]
swap_step_mult_list = [1, 2]
total_sims = len(sep_ratio_list)*len(beta_0_list)*len(proba_coeff_list)*len(batch_size_list)*len(burn_in_period_list)*len(swap_step_mult_list)

# simulate
timer = s_utils.Timer()
sim_num = 0
for batch_size in batch_size_list:
    for swap_step_mult in swap_step_mult_list:
        for beta_0 in beta_0_list:
            for proba_coeff in proba_coeff_list:
                for burn_in_period in burn_in_period_list:
                    for sep_ratio in sep_ratio_list:
                        sim_num += 1
                        train_data, train_labels, test_data, test_labels, valid_data, valid_labels = (
                                read_datasets.get_mnist_data())
                        swap_step = (100*swap_step_mult if batch_size == 500 else 1000*swap_step_mult)
                        noise_list = [beta_0*sep_ratio**i for i in range(n_replicas)]
                        test_step = (100 if batch_size == 500 else 900)
                        
                        name = s_utils.generate_experiment_name(model_name=func_name,
                                                                dataset_name=dataset_name,
                                                                separation_ratio=sep_ratio,
                                                                n_replicas=n_replicas,
                                                                beta_0=beta_0,
                                                                loss_func_name=loss_func_name,
                                                                swap_step=swap_step,
                                                                burn_in_period=burn_in_period,
                                                                learning_rate=learning_rate,
                                                                n_epochs=n_epochs,
                                                                noise_type=noise_type,
                                                                batch_size=batch_size,
                                                                proba_coeff=proba_coeff,
                                                                train_data_size=train_data_size)

                        print(name)
                        sim = Simulator(model=model,
                                        learning_rate=learning_rate,
                                        noise_list=noise_list,
                                        noise_type=noise_type,
                                        batch_size=batch_size,
                                        n_epochs=n_epochs,
                                        name=name,
                                        burn_in_period=burn_in_period,
                                        n_simulations=n_simulations,
                                        test_step=test_step,
                                        loss_func_name=loss_func_name,
                                        proba_coeff=proba_coeff)
                        sim.train_n_times(train_data_size=train_data_size,
                                          train_data=train_data,
                                          train_labels=train_labels,
                                          validation_data=valid_data,
                                          validation_labels=valid_labels,
                                          test_data=test_data,
                                          test_labels=test_labels)
                        print()
                        print(str(sim_num) + '/' + str(total_sims), ', time took:', timer.elapsed_time())

```