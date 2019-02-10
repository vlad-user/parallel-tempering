# Parallel Tempering

### Requirements

* [Python >= 3.5](https://www.python.org/)
* [TensorFlow 1.9.0 or 1.12.0](https://www.tensorflow.org/).
* [scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](http://www.numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyLatex](https://jeltef.github.io/PyLaTeX/current/)

## Examples

### 1. Simulate MNIST dataset with multilayer perceptron using Langevin dynamics optimizer
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

### 2. Simulate CIFAR10 using Lenet-5 with dropout with swaps, without swaps and using SGD with single replica and generate pdf report:

```python

import os

import tensorflow as tf
import numpy as np

from simulator import read_datasets
from simulator.simulator import Simulator
from simulator.summary_extractor import SummaryReportGenerator
from simulator.models.cifar10_models import lenet1_dropout
from simulator.models.cifar10_models import lenet1
from simulator import simulator_utils as s_utils

# hyper-parameters
n_epochs = 2000
batch_size = 50
beta_0 = 0.95
beta_n = 0.5
learning_rate = 0.01
proba_coeff = 50
loss_func_name = 'cross_entropy'
train_data_size = 7000
params = dict(
        n_replicas=[6, 6, 1],
        noise_type=['dropout_gd', 'dropout_gd', 'gd_no_noise'],
        model=[lenet1_dropout, lenet1_dropout, lenet1],
        burn_in_period=[2000, np.inf, np.inf],
        swap_step=[300, 2000, 2000],
        names=['swap', 'noswap', 'sgd'])

test_step = 800
timer = s_utils.Timer()
timer.start_timer()

# run simulations
train_data, train_labels, test_data, test_labels, valid_data, valid_labels = (
                                    read_datasets.get_cifar10_data())
for i in range(3):
    noise_list = list(np.linspace(beta_0, beta_n, params['n_replicas'][i]))

    sim = Simulator(model=params['model'][i],
                    learning_rate=learning_rate,
                    noise_list=noise_list,
                    noise_type=params['noise_type'][i],
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    name=params['names'][i],
                    burn_in_period=params['burn_in_period'][i],
                    swap_step=params['swap_step'][i],
                    separation_ratio=0,
                    n_simulations=1,
                    test_step=test_step,
                    loss_func_name=loss_func_name,
                    proba_coeff=proba_coeff
                    )
    
    sim.train(train_data_size=train_data_size,
              train_data=train_data,
              train_labels=train_labels,
              validation_data=valid_data,
              validation_labels=valid_labels,
              test_data=test_data,
              test_labels=test_labels)
    print()
    print(params['names'][i], ':', 'time took:', timer.elapsed_time())

report = SummaryReportGenerator(names=params['names'],
                                labels=params['names'],
                                report_name='my_report',
                                lower=1000,
                                higher=1500)
report.generate_report() # the report is in '/simulator/summaries/reports/my_report'

```