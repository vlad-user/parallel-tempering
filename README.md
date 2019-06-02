# Parallel Tempering

### Requirements

* [3.5 <= Python <= 3.6](https://www.python.org/)
* [TensorFlow 1.12.0](https://www.tensorflow.org/).
* [scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](http://www.numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyLatex](https://jeltef.github.io/PyLaTeX/current/)
* [Seaborn](https://seaborn.pydata.org/)

### Reproduction of results
To reproduce the plots from the paper follow instruction at `paper_results` subdirectory.

## Examples

### 1. Simulate CIFAR-10 using Lenet-5 architecture and compare between fixed varying dropout

```python
import os
import sys
import gc

import numpy as np

from simulator import read_datasets
from simulator.simulator import Simulator
from simulator.summary_extractor import SummaryReportGenerator
from simulator.models.cifar10_models_v2 import lenet5_with_dropout
from simulator import simulator_utils as s_utils


n_epochs = 400
learning_rate = 0.01
n_replicas = 8
noise_type = 'dropout_gd'
dataset_name = 'cifar'
loss_func_name = 'cross_entropy'
train_data_size = 45000
model = lenet5_with_dropout
model_name = 'lenet5'
description = 'testing'
beta_0 = .45
beta_n = .53
proba_coeff = 2000
batch_size = 128
burn_in_period_list = [10000, np.inf] # <-- simulate with and without swaps
swap_step = 600
mode = None 
test_step = 352 # <-- number of steps between computations of test error
timer = s_utils.Timer()

x_train, y_train, x_test, y_test, x_valid, y_valid = (
    read_datasets._create_cifar_data_or_get_existing_lenet5())

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
  print(log)

# generate pdf report
report_name = 'lenet5-dropout'
labels = ['Swaps', 'No Swaps']
repgen = SummaryReportGenerator(names,
                                labels=labels,
                                lower=100,
                                higher=200,
                                ylim_err=(0.2, 0.35),
                                ylim_loss=(0, 3.))
repgen.generate_report()
# Now open `parallel-tempring/simulator/summaries/reports/lenet5-dropout/lenet5-dropout.pdf`
```
