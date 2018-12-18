import os
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt

from simulator.plot import Plot


class SummaryExtractor:

  def __init__(self, name):
    dirname = os.path.abspath(os.path.dirname(__file__))
    self._dirname = os.path.join(dirname, 'summaries', name)
    filenames = sorted([f for f in os.listdir(self._dirname) if 'summary' in f],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))
    description_path = os.path.join(self._dirname, 'description.json')
    self._summaries = []
    self._n_simulations = len(filenames)
    for f in filenames:
      with open(os.path.join(self._dirname, f), 'rb', os.O_NONBLOCK) as fo:
        self._summaries.append(pickle.load(fo))

    with open(description_path, 'r') as fo:
      self._description = json.load(fo)

    self._vals = {
        'accept_ratio': None,
        'mix_ratio': None,
        'visit_ratio': None
    }

    self._n_replicas = self.get_description()['n_replicas']

  def show_report(self, simulation_num=0, sample_every=1):
    print('Accept Ratio:', self.get_accept_ratio())
    print('Visit Ratio:', self.get_visit_ratio())
    print('Mixing Ratio:', self.get_mix_ratio())
    print()
    _ = self._plot_loss(simulation_num=simulation_num)
    _ = self._plot_loss(summ_name='train_loss',
                        simulation_num=simulation_num,
                        sample_every=sample_every)
    _ = self._plot_error(simulation_num=simulation_num)
    _ = self._plot_error(summ_name='train_error',
                         simulation_num=simulation_num,
                         sample_every=sample_every)
    _ = self._plot_diffusion(simulation_num=simulation_num)
    _ = self._plot_mixing(simulation_num=simulation_num)
    _ = self._plot_grads(simulation_num=simulation_num)
    _ = self._plot_norms(simulation_num=simulation_num)


  def get_accept_ratio(self):
    accepts = []
    if self._vals['accept_ratio'] is None:
      for s in range(self._n_simulations):
        for r in range(self._n_replicas):
          x, y = self.get_summary('accepts', replica_id=r, simulation_num=s)
          accepts.append(np.mean(y))
      self._vals['accept_ratio'] = np.mean(accepts)

    return self._vals['accept_ratio']

  def get_mix_ratio(self):
    if self._vals['mix_ratio'] is None:
      keys = [float("{0:.4f}".format(b))
              for b in self.get_description()['noise_list']]

      def _get_key(key):
        return keys[int(np.argmin([abs(k-key) for k in keys]))]

      mixing = {i:[] for i in range(self._n_replicas)}
      visiting = {i:[] for i in range(self._n_replicas)}
      for s in range(self._n_simulations):

        for r in range(self._n_replicas):
          x, y = self.get_summary('noise_values', replica_id=r, simulation_num=s)
          steps = self.get_summary('train_steps', replica_id=r, simulation_num=s)

          reps = {k:0 for k in keys}

          for i in range(len(steps[1])):
            if steps[1][i] > self.get_description()['burn_in_period']:
              reps[_get_key(y[i])] += 1

          visiting[r].append(np.mean([1 if reps[x]!=0 else 0 for x in reps]))
          mixing[r].append(1 if all(reps[x]!=0 for x in reps) else 0)

      mix_ratios = []
      visit_ratios = []
      for s in range(self._n_simulations):
        mix_ratio = np.mean([mixing[r][s] for r in range(self._n_replicas)])
        visit_ratio = np.mean([visiting[r][s] for r in range(self._n_replicas)])
        mix_ratios.append(mix_ratio)
        visit_ratios.append(visit_ratio)
      self._vals['mix_ratio'] = np.mean(mix_ratios)
      self._vals['visit_ratio'] = np.mean(visit_ratios) * self._n_replicas

    return self._vals['mix_ratio']

  def get_visit_ratio(self):
    if self._vals['visit_ratio'] is None:
      _ = self.get_mix_ratio()
    return self._vals['visit_ratio']


  def get_summary(self, summ_name, replica_id=0, simulation_num=0):
    if simulation_num >= self._n_simulations:
      raise ValueError('No such simulation.')
    if 'steps' in summ_name:
      y = self._summaries[simulation_num][summ_name]
    else:
      y = self._summaries[simulation_num][summ_name][replica_id]
    n_epochs = self._summaries[simulation_num]['latest_epoch'] + 1
    try:
      x = np.linspace(start=0,
                      stop=n_epochs,
                      num=len(y))
    except TypeError:
      print(summ_name, y)
      raise
    return x, y

  def get_description(self):
    return self._description

  def _plot_norms(self, simulation_num=0):
    fig, ax = plt.subplots()
    plot = Plot()
    for r in range(self._n_replicas):
      x, y = self.get_summary('weight_norms', r, simulation_num)
      plot.plot(x, y, fig=fig, ax=ax, label='replica ' + str(r),
                linewidth=2)
    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='WEIGHT L2 NORM')
    return fig

  def _plot_diffusion(self, simulation_num=0):
    fig, ax = plt.subplots()
    plot = Plot()
    for r in range(self._n_replicas):
      x, y = self.get_summary('diffusion', r, simulation_num)
      plot.plot(x, y, fig=fig, ax=ax, label='replica ' + str(r),
                linewidth=2)

    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='DIFFUSION')
    return fig

  def _plot_grads(self, simulation_num=0):
    fig, ax = plt.subplots()
    plot = Plot()
    for r in range(self._n_replicas):
      x, y = self.get_summary('grad_norms', r, simulation_num)
      plot.plot(x, y, fig=fig, ax=ax, label='replica ' + str(r),
                linewidth=1.5)

    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='GRADIENT L2 NORM', log_y=5)

    return fig

  def _plot_loss(self, summ_name='test_loss', simulation_num=0, sample_every=1):
    fig, ax = plt.subplots()
    plot = Plot()
    for r in range(self._n_replicas):
      x, y = self.get_summary(summ_name, r, simulation_num)
      x, y = x[::sample_every], y[::sample_every]
      plot.plot(x, y, fig=fig, ax=ax, label='replica ' + str(r),
                linewidth=2, splined_points_mult=None)

    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='LOSS', ylimit=(0, 9))

  def _plot_error(self, summ_name='test_error', simulation_num=0, sample_every=1):
    fig, ax = plt.subplots()
    plot = Plot()
    for r in range(self._n_replicas):
      x, y = self.get_summary(summ_name, r, simulation_num)
      x, y = x[::sample_every], y[::sample_every]
      label = 'replica_' + str(r) + ': min_err=' + "{0:.2f}".format(min(y))
      plot.plot(x, y, fig=fig, ax=ax, label=label,
                linewidth=2, splined_points_mult=None)

    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='0-1 ERROR')

  def _plot_mixing(self, simulation_num=0):
    def _get_key(key):
      keys = [float("{0:.4f}".format(b))
              for b in self.get_description()['noise_list']]
      return keys[int(np.argmin([abs(k-key) for k in keys]))]

    fig, ax = plt.subplots()
    plot = Plot()
    noise_list = self.get_description()['noise_list']

    key_map = {_get_key(key):i for i, key in enumerate(noise_list)}
    for r in range(self._n_replicas):
      x, y = self.get_summary('noise_values', replica_id=r, simulation_num=simulation_num)

      y_new = [key_map[_get_key(i)] for i in y]
      plot.plot(x, y_new, fig=fig, ax=ax, label='replica ' + str(r),
                linewidth=2)
    yticks_names = [float("{0:.4f}".format(b)) for b in noise_list]

    plt.gca().set_yticklabels(['0'] + yticks_names)
    plot.legend(fig, ax, legend_title='ReplicaID',
                xlabel='EPOCHS', ylabel='NOISE LEVEL')
    return fig