"""Downloads and returns cifar10 dataset."""
import os
import tarfile
import random
import sys
from six.moves import urllib
import urllib.request
import pickle
import gzip
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset
import tensorflow as tf

EMNIST_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

class Cifar10: # pylint: disable=too-many-instance-attributes, too-few-public-methods, missing-docstring

  def __init__(self, batch_size=50, data_url=None, data_dirname='cifar10'):

    self.DATA_URL = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                    if data_url is None else data_url)
    self.data_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/data/' # pylint: disable=line-too-long
    self.cifar10_dir = os.path.join(self.data_dir, data_dirname)
    
    self.batch_size = batch_size
    self._download_and_extract()
    self._prepare_dataset()

  def _prepare_dataset(self):
    """
    # PYTHON 2.7.x
    import cPickle as pickle
    def unpickle(file):

      with open(file , 'rb') as fo:
        res = pickle.load(fo)
      return res
    """
    # PYTHON 3.x.x
    def unpickle(file):
      with open(file, 'rb') as file_:
        unpick = pickle._Unpickler(file_) # pylint: disable=protected-access
        unpick.encoding = 'latin1'
        res = unpick.load()
      return res

    def extract_data(datatype):
      """Returns tuple numpy arrays of data, labels

      Args:
        `datatype`: A string, 'test', 'train', or 'validation'
      """
      batches = []
      if datatype == 'train':
        str2search = 'batch_'
      elif datatype == 'test':
        str2search = 'test'
      elif datatype == 'validation':
        str2search = 'test'

      for file in os.listdir(self.cifar10_dir):
        file_path = os.path.join(self.cifar10_dir, file)
        if os.path.isfile(file_path) and str2search in file:
          batches.append(unpickle(file_path))
      data = np.concatenate(tuple(a['data'] for a in batches))
      labels = np.concatenate(tuple(a['labels'] for a in batches))
      return data, labels

    self.train_data, self.train_labels = extract_data('train')
    self.test_data, self.test_labels = extract_data('test')
    self.valid_data, self.valid_labels = extract_data('validation')

  def _download_and_extract(self):
    """Download from https://www.cs.toronto.edu/~kriz/cifar.html if the
    the file is not located in the path"""
    if not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)
    dest_directory = self.cifar10_dir

    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)

    filename = self.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)


    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(
          self.DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    self.cifar10_dir = extracted_dir_path

def get_cifar10_data(validation_size=0.5, random_state=None):
  """Returns cifar10 data. If not on disk, downloads."""
  # cif = Cifar10()
  # X_test, y_test = cif.test_data, cif.test_labels # pylint: disable=invalid-name
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_test, x_valid, y_test, y_valid = train_test_split( # pylint: disable=invalid-name
      x_test,
      y_test, test_size=validation_size,
      random_state=(random_state
                    if random_state is not None else random.randint(1, 42)))
  #X_train = cif.train_data
  #y_train = cif.train_labels
  y_train, y_test, y_valid = (y_train.flatten(),
                              y_test.flatten(),
                              y_valid.flatten())
  return x_train/255, y_train, x_test/255, y_test, x_valid/255, y_valid

def get_cifar10_data_debug(train_data_size=4000, replace_existing=False):
  """Creates (if not exists) and returns cifar data for debug."""

  dirname = os.path.join(os.path.dirname(__file__), 'data', 'debug_cifar10')
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  fname = os.path.join(dirname, 'data.pkl')
  
  if not os.path.exists(fname) or replace_existing:
    x_train, y_train, x_test, y_test, x_valid, y_valid = (
        get_cifar10_data())

    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, y_train = x_train[:train_data_size], y_train[:train_data_size]

    data = {
      'x_train': x_train,
      'y_train': y_train,
      'x_test': x_test,
      'y_test': y_test,
      'x_valid': x_valid,
      'y_valid': y_valid
    }
    with open(fname, 'wb') as fo:
      pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

  with open(fname, 'rb') as fo:
    data = pickle.load(fo)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_valid = data['x_valid']
    y_valid = data['y_valid']

  return x_train, y_train, x_test, y_test, x_valid, y_valid



def get_fashion_mnist_data(validation_size=0.1, random_state=None, flatten=True):
  """Returns fashion mnist dataset. If not on disk, downloads."""
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
  if flatten:

    shape = X_train.shape
    X_train = np.reshape(X_train, (shape[0], shape[1]*shape[2]))
    shape = X_test.shape
    X_test = np.reshape(X_test, (shape[0], shape[1]*shape[2]))

  X_test, X_valid, y_test, y_valid = train_test_split(
      X_test, y_test, test_size=0.5, random_state=(random_state
          if random_state is not None else random.randint(1, 42)))

  return X_train, y_train, X_test, y_test, X_valid, y_valid

def get_mnist_data(validation_size=0.5, random_state=None, flatten=True):
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  if flatten:

    shape = X_train.shape
    X_train = np.reshape(X_train, (shape[0], shape[1]*shape[2]))
    shape = X_test.shape
    X_test = np.reshape(X_test, (shape[0], shape[1]*shape[2]))

  X_test, X_valid, y_test, y_valid = train_test_split(
      X_test, y_test, test_size=0.5, random_state=(random_state
          if random_state is not None else random.randint(1, 42)))

  return X_train/255, y_train, X_test/255, y_test, X_valid/255, y_valid

def _create_mnist_data_or_get_existing(fname='mnist.pkl'):
  
  dirname = os.path.dirname(__file__)
  dirname = os.path.join(dirname, 'data')

  if not os.path.exists(dirname):
    os.makedirs(dirname)
  fname = os.path.join(dirname, fname)
  if os.path.exists(fname):
    with open(fname, 'rb') as fo:
      obj = pickle.load(fo)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    x_valid = obj['x_valid']
    y_valid = obj['y_valid']

  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_test = x_train[..., None], x_test[..., None]
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=5./ 60.)

    mean = np.mean(x_train, axis=0)
    x_train = x_train - mean
    x_test = x_test - mean
    x_valid = x_valid - mean
    
    with open(fname, 'wb') as fo:
      obj = {
        'x_train': x_train, 
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_valid': x_valid,
        'y_valid': y_valid
      }
      pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
    
  return x_train, y_train, x_test, y_test, x_valid, y_valid

def _create_cifar_data_or_get_existing_resnet(fname='resnet_cifar.pkl'):
  # store exact same split for feature simulations or load already existing
  dirname = os.path.dirname(__file__)
  dirname = os.path.join(dirname, 'data')


  if not os.path.exists(dirname):
    os.makedirs(dirname)
  fname = os.path.join(dirname, fname)
  if os.path.exists(fname):
    with open(fname, 'rb') as fo:
      obj = pickle.load(fo)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    x_valid = obj['x_valid']
    y_valid = obj['y_valid']
  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    x_train, x_test = x_train / 255., x_test / 255.

 
    x_data = np.concatenate((x_train, x_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    
    x_data, y_data = shuffle_dataset(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=1/6,
                                                        shuffle=True)
    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.1)
    # per-pixel mean subtraction
    mean = np.mean(x_train, axis=0)[None, ...]
    x_train = x_train - mean
    x_test = x_test - mean
    x_valid = x_valid - mean

    with open(fname, 'wb') as fo:
      obj = {
        'x_train': x_train, 
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_valid': x_valid,
        'y_valid': y_valid
      }
      pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
  return x_train, y_train, x_test, y_test, x_valid, y_valid

def _create_cifar_data_or_get_existing_lenet5():
  # store exact same split for feature simulations or load already existing
  return _create_cifar_data_or_get_existing_resnet()
  dirname = os.path.dirname(__file__)
  dirname = os.path.join(dirname, 'data')
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  fname = os.path.join(dirname, 'lenet_cifar.pkl')
  if os.path.exists(fname):
    with open(fname, 'rb') as fo:
      obj = pickle.load(fo)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    x_valid = obj['x_valid']
    y_valid = obj['y_valid']
  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    with open(fname, 'wb') as fo:
      obj = {
        'x_train': x_train, 
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_valid': x_valid,
        'y_valid': y_valid
      }
      pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
  return x_train, y_train, x_test, y_test, x_valid, y_valid

def get_emnist_letters(fname='emnist-letters-from-src.pkl'):

  _maybe_download_emnist()
  dirname = os.path.dirname(os.path.abspath(__file__))
  dirname = os.path.join(dirname, 'data')
  fname = os.path.join(dirname, fname)
  if os.path.exists(fname):
    with open(fname, 'rb') as fo:
      obj = pickle.load(fo)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    x_valid = obj['x_valid']
    y_valid = obj['y_valid']


  else:
    gzip_path = os.path.dirname(os.path.abspath(__file__))
    gzip_path = os.path.join(gzip_path, 'data')
    dst_path = os.path.join(gzip_path, 'emnist_extracted')
    gzip_path = os.path.join(gzip_path, 'gzip.zip')
    
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
    dst_path = os.path.join(dst_path, 'gzip', 'gzip')
    fnames_dict = {
      'x_test': 'emnist-letters-test-images-idx3-ubyte.gz',
      'y_test': 'emnist-letters-test-labels-idx1-ubyte.gz',
      'x_train': 'emnist-letters-train-images-idx3-ubyte.gz',
      'y_train': 'emnist-letters-train-labels-idx1-ubyte.gz'
      }
    fullpaths = {k: os.path.join(dst_path, v) for k, v in fnames_dict.items()}
    
    for attempt in range(5):
      try:
        zip_ref = zipfile.ZipFile(gzip_path)
        zip_ref.extractall(dst_path)
        zip_ref.close()
        break
      except zipfile.BadZipFile:
        
        if attempt == 4:
          err_msg = ("Can't download EMNIST dataset. Try "
                     "downloading EMNIST dataset manually and place the "
                     "gzip.zip file to the parallel-tempring/simulator/data "
                     "folder.")
          raise ValueError(err_msg)
        os.remove(gzip_path)
        _maybe_download_emnist()

    def _read4bytes(bytestream):
      dtype = np.dtype(np.uint32).newbyteorder('>')
      return np.frombuffer(bytestream.read(4), dtype=dtype)[0]

    def ungzip_data(fname):
      with gzip.GzipFile(fname, 'r') as fo:
        magic = _read4bytes(fo)
        n_images = _read4bytes(fo)
        n_rows = _read4bytes(fo)
        n_cols = _read4bytes(fo)
        buf = fo.read()
        data = np.frombuffer(buf, dtype=np.uint8)
      return data.reshape(n_images, n_rows, n_cols, 1)

    def ungzip_labels(fname):
      with gzip.GzipFile(fname, 'r') as fo:
        magic = _read4bytes(fo)
        n_labels = _read4bytes(fo)
        buf = fo.read()
        data = np.frombuffer(buf, dtype=np.uint8)
      return data

    x_train = ungzip_data(fullpaths['x_train'])
    y_train = ungzip_labels(fullpaths['y_train'])
    x_test = ungzip_data(fullpaths['x_test'])
    y_test = ungzip_labels(fullpaths['y_test'])



    x_train = np.float32(x_train) / 255.
    x_test = np.float32(x_test) / 255.
    y_train = np.int32(y_train) - 1
    y_test = np.int32(y_test) - 1

    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.1)

    with open(fname, 'wb') as fo:
      obj = {
        'x_train': x_train, 
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_valid': x_valid,
        'y_valid': y_valid
      }
      pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)

  return x_train, y_train, x_test, y_test, x_valid, y_valid

def _maybe_download_emnist():

  filepath = os.path.dirname(os.path.abspath(__file__))
  filepath = os.path.join(filepath, 'data')
  if not os.path.exists(filepath):
    os.makedirs(filepath)
  filename = EMNIST_URL.split('/')[-1]
  filepath = os.path.join(filepath, filename) 
  
  if os.path.exists(filepath):
    return
  def _progress(count, block_size, total_size):
    buff = '\r>> Downloading EMNIST %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0)
    sys.stdout.write(buff)
    sys.stdout.flush()

  filepath, _ = urllib.request.urlretrieve(
    EMNIST_URL, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded EMNIST dataset', statinfo.st_size, 'bytes.')

if __name__ == '__main__':

  _ = get_emnist_letters()