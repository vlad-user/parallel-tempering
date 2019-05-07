"""Helper model for assigning ops to devices.
'export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/'
'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/lib/nvidia-418/'
"""
from __future__ import absolute_import
import os

from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

from simulator.exceptions import NoGpusFoundError

RAISE_IF_NO_GPU = True

# **NOTE**: for different systems this path may differ. 
#os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64/'

def _get_available_gpus():
  """Returns a list of available gpu names."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _gpu_device_name(replica_id):
  """Returns a name of the device for `replica_id`"""
  gpus = _get_available_gpus()
  if RAISE_IF_NO_GPU and not gpus:
    raise NoGpusFoundError()
  if not gpus:
    return '/cpu:0'
  return '/gpu:' + str(replica_id % len(gpus))
