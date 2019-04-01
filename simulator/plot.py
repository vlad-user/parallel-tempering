from math import isinf
import numpy as np

import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import spline


class Plot:
  """Helper for plotting."""
  def __init__(self):
    self.__first_use = True

  def legend(self,
             fig,
             ax, # pylint: disable=invalid-name
             bbox_to_anchor=(1.1, 0.5),
             legend_title='',
             xlabel=None,
             ylabel=None,
             title=None,
             log_x=None,
             log_y=None,
             fig_width=9,
             fig_height=4,
             ylimit=None,
             xlimit=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
    ax.legend(
        loc='center right', fancybox=True, shadow=True,
        bbox_to_anchor=bbox_to_anchor, title=legend_title)

    if title is not None:
      ax.title.set_text(title)
    if xlabel is not None:
      ax.set_xlabel(xlabel)
    if ylabel is not None:
      ax.set_ylabel(ylabel)
    if log_x is not None:
      plt.xscale('log', basex=log_x)
    if log_y is not None:
      plt.yscale('log', basey=log_y)
    if ylimit is not None:
      plt.ylim(ylimit[0], ylimit[1])
    if xlimit is not None:
      plt.xlim(xlimit[0], xlimit[1])

    fig.set_size_inches(fig_width, fig_height) # (width, height)
    self.__first_use = False

  def plot( # pylint:disable=invalid-name
      self, x, y, err=None, fig=None, ax=None, label=None,
      color=None, linestyle='-', ylim_top=None, ylim_bottom=None, 
      splined_points_mult=None, linewidth=None, elinewidth=0.5,
      markeredgewidth=0.05, capsize=1, fmt=None, marker=None,
      remove_discontinuity=None, annotate=None):
    """
    Args:
      annotate: If not None, adds the annotation to each point in plot.
        Must be a list of triples (txt, x_val, y_val)
      marker: The marker of each point. 
    """

    def max_(array):
      list_ = [a for a in array if isinf(a) is False]
      return max(list_)

    if fig is None or ax is None:
      fig, ax = plt.subplots()

    if err is not None:
      plot_func = ax.errorbar
    else:
      plot_func = ax.plot

    if color is None and fmt is not None:
      color = fmt[0]
    # check if there are infinity vals and replace by finite "big" vals
    x_ = [e  if isinf(e) is False else max_(x) + max_(x)*2 # pylint:disable=invalid-name
          for e in x]
    y_ = [e if isinf(e) is False else max_(y) + max_(y)*2 # pylint:disable=invalid-name
          for e in y]

    x = np.array(x_) # pylint:disable=invalid-name
    y = np.array(y_) # pylint:disable=invalid-name

    if splined_points_mult is not None:
      x_new = np.linspace(x.min(), x.max(), x.shape[0]*splined_points_mult)
      y_new = spline(x, y, x_new)
      for i in range(y_new.shape[0]):
        if y_new[i] < 0:
          y_new[i] = 0.0
      if err:
        err_new = spline(x, err, x_new)
        plot_func(
            x_new, y_new, yerr=err_new,
            errorevery=x_new.shape[0]/splined_points_mult,
            label=label, elinewidth=elinewidth,
            markeredgewidth=markeredgewidth,
            capsize=capsize, linestyle=linestyle,
            marker=marker)
      else:
        if fmt is None:

          plot_func(x_new, y_new, color=color, label=label,
                    linewidth=linewidth, linestyle=linestyle,
                    marker=marker)
        else:
          plot_func(x_new, y_new, fmt, color=color, label=label,
                    linewidth=linewidth, linestyle=linestyle,
                    marker=marker)

    else:
      if remove_discontinuity is not None:
        position = np.where(np.abs(np.diff(y)) >= remove_discontinuity)[0]

        x[position] = np.nan
        y[position] = np.nan
      if err:
        if fmt is None:
          plot_func(x, y, yerr=err, color=color, label=label, linewidth=linewidth,
                    capsize=capsize, markeredgewidth=markeredgewidth,
                    elinewidth=elinewidth, linestyle=linestyle,
                    marker=marker)
          
        else:
          plot_func(x, y,yerr=err, fmt=fmt, color=color, label=label,
                    linewidth=linewidth, markeredgewidth=markeredgewidth,
                    capsize=capsize, elinewidth=elinewidth, linestyle=linestyle,
                    marker=marker)

      else:

        if fmt is None:

          plot_func(x, y, label=label, color=color, linewidth=linewidth,
                    linestyle=linestyle,marker=marker)

        else:
          plot_func(x, y, fmt, color=color, label=label, linewidth=linewidth,
                    linestyle=linestyle, marker=marker)
    if annotate is not None:
      for x in annotate:
        ax.annotate(x[0], (x[1], x[2]))



    return fig

# helper functions to plot different distributions
import math

def rbf_kernel(x1, x2, alpha, radius, sigma):
  return (sigma**2)*math.exp(-1 * np.power(abs(x1-x2), alpha) / (2*radius**2))

def plot_gaussian_prior(n_functions=2, kernel=rbf_kernel,
                        radius=1, mean=0, sigma=2, alpha=2,
                        range_=(-1, 1), n_points=200,):
  """Generates functions drawn from Gaussian Process prior.

  Args:
    n_functions: A number of functions to plot.
    kernel: A covariance function.
    radius: A radius of the kernel.
    range_: A range for `x` axis.
    sigma: A scaling of the kernel.
    n_points: A number of points to generate.
  """
  xs = np.linspace(range_[0], range_[1], n_points)
  mean = mean*np.ones_like(xs)
  gram = gram_matrix(xs, alpha=alpha, radius=radius, sigma=sigma)
  plt_vals = []

  fig, ax = plt.subplots()
  plot = Plot()
  for i in range(n_functions):
    ys = np.random.multivariate_normal(mean, gram)
    ys = ys - (ys.min() if ys.min() < 0 else 0)
    ys = ys/ys.max()
    plt_vals.append([xs, ys, 'k'])

  for i, x in enumerate(plt_vals):
    plot.plot(x[0], x[1], fig=fig, ax=ax, label=str(i+1))
  plot.legend(fig=fig, ax=ax, )



def gram_matrix(xs, kernel=rbf_kernel, alpha=2, radius=1, sigma=2):
  return [[kernel(x1,x2, alpha, radius, sigma) for x2 in xs] for x1 in xs]