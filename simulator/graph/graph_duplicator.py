"""Module for duplicating tensorflow graphs connected to the same inputs.
This is a modified version of tf.contrib.copy_graph

**NOTE**: `tf.layers.batch_normalization` is not working properly after
being copied. 
"""
import sys
from copy import deepcopy
import random
from distutils.version import StrictVersion

from tensorflow.python.framework import ops # pylint: disable=no-name-in-module
import tensorflow as tf
import numpy as np

from simulator.graph.device_placer import _gpu_device_name

COPIED_VARIABLES = {}

def random_uniform(interval1=(0.8, 1.2), interval2=(-1.2, -0.8)):
  if np.random.uniform() >= 0.5:
    return np.random.uniform(low=interval1[0], high=interval1[1])
  else:
    return np.random.uniform(low=interval2[0], high=interval2[1])

def copy_and_duplicate( # pylint: disable=too-many-arguments, invalid-name
    X, y, is_train, logits, n_duplicates, dst_graph,
    dropout_placeholder=None, namespace='Layers_'):
  """Duplicates a graph `n_duplicates` times.

  A `dst_graph` will contain the duplicates of the net, except that
  `X`, `y` and `is_train` placeholders are copied only once and connect
  to all duplicated nets.

  Args:
    `X`: Placeholder of existing graph.
    `y`: Placeholder of existing graph.
    `is_train`: Placeholder indicating whether a model is in execution
      or inference mode.
    `logits`: logits layer of existing graph
    `n_duplicates`: number of copies to create
    `dst_graph`: graph to which to copy
    `dropout_placeholder`: a placeholder for keep prob.

  Returns:
    A tuple `(X, y, is_train, logits_list)` if
    `dropout_placeholders is None`, or a tuple 
    `(X, y, is_train, dropout_placeholders, logits_list)` otherwise.

  """

  X_copy = copy_to_graph(X, dst_graph, namespace='') # pylint: disable=invalid-name
  y_copy = copy_to_graph(y, dst_graph, namespace='') # pylint: disable=invalid-name
  is_train_copy = copy_to_graph(is_train, dst_graph, namespace='')

  for i in range(n_duplicates):
    _ = [copy_variable_to_graph(v, dst_graph, namespace=namespace+str(i))
         for v in X.graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)]

  logits_list = [copy_to_graph(
      logits,
      dst_graph,
      namespace=namespace + str(i),
      exclude=[X_copy, y_copy, is_train_copy]) for i in range(n_duplicates)]


  if dropout_placeholder is not None:
    copied_dropout_placeholders = \
      [get_copied(dropout_placeholder, dst_graph, namespace=namespace+str(i))
       for i in range(n_duplicates)]

    return X_copy, y_copy, is_train_copy, copied_dropout_placeholders, logits_list

  return X_copy, y_copy, is_train_copy, logits_list


def copy_variable_to_graph(org_instance, to_graph, namespace):
  """Copies a variable `org_instance` to the graph `to_graph`.

  Args:
    `org_instance`: Original variable instance.
    `to_graph`: Destination graph to where the variable will be
      copied.
      `namespace`: A new namespace under which the variable will
      be copied.

  Returns:
    A copied variable.

  Raises:
    `TypeError`: If `org_instance` is not an instance of `tf.Variable`.
  """

  if not isinstance(org_instance, tf.Variable):
    raise TypeError(str(org_instance) + " is not a Variable")

  #The name of the new variable
  if namespace != '':
    new_name = (namespace + '/' +
                org_instance.name[:org_instance.name.index(':')])
  else:
    new_name = org_instance.name[:org_instance.name.index(':')]

  ###############################################################
  if namespace != '':
    replica_id = int(namespace.split('_')[1])
  else:
    replica_id = -1
  ###############################################################

  # Get the collections that the new instance needs to be added to.
  # The new collections will also be a part of the given namespace,
  # except the special ones required for variable initialization and
  # training.
  collections = []
  for name, collection in org_instance.graph._collections.items(): # pylint: disable=protected-access
    if org_instance in collection:
      if (name == ops.GraphKeys.GLOBAL_VARIABLES or
          name == ops.GraphKeys.TRAINABLE_VARIABLES or
          namespace == ''):
        collections.append(name)
      else:
        collections.append(namespace + '/' + name)

  # See if its trainable.
  trainable = (org_instance in org_instance.graph.get_collection(
      ops.GraphKeys.TRAINABLE_VARIABLES))

  # Get the initial value

  with org_instance.graph.as_default():
    temp_session = tf.Session() # pylint: disable=unused-variable
    init_value = temp_session.run(org_instance.initialized_value())
    init_value *= random_uniform()

  # Initialize the new variable
  with to_graph.as_default():
    new_var = tf.Variable(init_value,
                trainable=trainable,
                name=new_name,
                collections=collections,
                validate_shape=True)
    
  # Add to the COPIED_VARIABLES dict
  COPIED_VARIABLES[new_var.name] = new_var

  return new_var

def copy_to_graph(org_instance, to_graph, namespace="", exclude=None):
  """Creates a copy of the `Operation`/`Tensor` `org_instance.

  The copying is done recursively with the help of `COPIED_VARIABLES`
  dictionary which stores already copied instances. Additionaly,
  it is possible to exclude additional `Tensor`s or `Operation`s by
  putting already copied instances to the `exclude` list. The copied
  instances are inserted under the provided `namespace`. To avoid
  naming conflicts it is better to provide a value for `namespace`.

  Args:
    `org_instance`: A instance of `Operation` or `Tensor` to be copied
      to the `to_graph`.
    `to_graph`: A graph where `org_instance` should be copied.
    `namespace`: A namespace under which the `org_instance` will be
      copied.
    `exclude`: A list of variables/tensors/ops that should not be
      copied.

  Returns:
    A copied instance.

  Raises:
    `ValueError`: If the intance couldn't be copied.
  """  
  ####################################################################
  if namespace != '':
    replica_id = int(namespace.split('_')[1])
  else:
    replica_id = -1
  global COPIED_VARIABLES # pylint: disable=global-statement
  if exclude:
    for exc in exclude:
      if org_instance.name.split(':')[0] == exc.name.split(':')[0]:
        return exc

  ####################################################################

  # The name of the new instance
  if namespace != '':
    new_name = namespace + '/' + org_instance.name
  else:
    new_name = org_instance.name

  # If a variable by the new name already exists, return the
  # correspondng tensor that will act as an input
  if new_name in COPIED_VARIABLES:
    print('copied',new_name)
    return to_graph.get_tensor_by_name(
        COPIED_VARIABLES[new_name].name)

  # If an instance of the same name exists, return appropriately
  try:
    already_present = to_graph.as_graph_element(
        new_name,
        allow_tensor=True,
        allow_operation=True)

    return already_present
  except: # pylint: disable=bare-except
    pass

  # Get the collections that the new instance needs to be added to.
  # The new collections will also be a part of the given namespace.
  collections = []
  for name, collection in org_instance.graph._collections.items(): # pylint: disable=protected-access
    if org_instance in collection:
      if namespace == '':
        collections.append(name)
      else:
        collections.append(namespace + '/' + name)

  # Take action based on the class of the instance

  if isinstance(org_instance, tf.Tensor): # pylint: disable=no-else-return

    # If its a Tensor, it is one of the outputs of the underlying
    # op. Therefore, copy the op itself and return the appropriate
    # output.
    op = org_instance.op # pylint: disable=invalid-name
    new_op = copy_to_graph(op, to_graph, namespace, exclude=exclude)
    output_index = op.outputs.index(org_instance)
    new_tensor = new_op.outputs[output_index]

    #Add to collections if any
    for collection in collections:
      to_graph.add_to_collection(collection, new_tensor)

    return new_tensor

  elif isinstance(org_instance, tf.Operation):
    op = org_instance # pylint: disable=invalid-name

    # If it has an original_op parameter, copy it
    if op._original_op is not None: # pylint: disable=protected-access
      new_original_op = copy_to_graph(
          op._original_op, to_graph, # pylint: disable=protected-access
          namespace, exclude=exclude)
    else:
      new_original_op = None

    # If it has control inputs, call this function recursively on each.
    new_control_inputs = [copy_to_graph(x, to_graph,
                                        namespace,
                                        exclude=exclude)
                          for x in op.control_inputs]

    # If it has inputs, call this function recursively on each.
    new_inputs = [copy_to_graph(x, to_graph,
                                namespace, exclude=exclude)
                  for x in op.inputs]

    # Make a new node_def based on that of the original.
    # An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
    # stores String-based info such as name, device and type of the op.
    # Unique to every Operation instance.
    new_node_def = deepcopy(op.node_def)
    # Change the name
    new_node_def.name = new_name

    # Copy the other inputs needed for initialization
    output_types = op._output_types[:] # pylint: disable=protected-access
    input_types = op._input_types[:] # pylint: disable=protected-access

    # Make a copy of the op_def too.
    # Its unique to every _type_ of Operation.
    op_def = deepcopy(op.op_def)

    # Initialize a new Operation instance

    new_op = tf.Operation(
        new_node_def,
        to_graph,
        new_inputs,
        output_types,
        new_control_inputs,
        input_types,
        new_original_op,
        op_def)
    
    ########################################################
    if StrictVersion(tf.__version__) > StrictVersion("1.9.0"): 
      to_graph._record_op_seen_by_control_dependencies(new_op)
      for device_function in to_graph._device_functions_outer_to_inner:
        new_op._set_device(device_function(new_op))
    else:
      to_graph._add_op(new_op)
      for device_function in reversed(to_graph._device_function_stack):
        new_op._set_device(device_function(new_op)) # pylint: disable=protected-access
        to_graph._record_op_seen_by_control_dependencies(new_op)
    
    if (replica_id >= 0 and
        'gpu' in op.device.lower()):
      new_op._set_device(_gpu_device_name(replica_id)) # pylint: disable=protected-access

    return new_op
    ########################################################

  else:
    raise ValueError("Could not copy instance: " + str(org_instance))


def get_copied(original, graph, namespace=""):
  """Returns a copy of the `original` that presents in `graph` under `namespace`."""

  # The name of the copied instance
  if namespace != '':
    new_name = namespace + '/' + original.name
  else:
    new_name = original.name

  # If a variable by the name already exists, return it
  if new_name in COPIED_VARIABLES:
    return COPIED_VARIABLES[new_name]

  return graph.as_graph_element(
      new_name, allow_tensor=True, allow_operation=True)
