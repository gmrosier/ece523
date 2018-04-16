#!/usr/bin/env python
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

class CifarDataset():
  def __init__(self, parent_dir, is_training=True):
    self._data_dir = os.path.join(parent_dir, 'cifar-10-batches-bin')
    self._training = is_training
    self._training_files = []
    for item in os.listdir(self._data_dir):
      if item.startswith('data_batch_'):
        self._training_files.append(os.path.join(self._data_dir, item))

    self._test_files = [os.path.join(self._data_dir, 'test_batch.bin')]

    self._catagories = []
    with open(os.path.join(self._data_dir, 'batches.meta.txt'), 'r') as fd:
      line = fd.readline()
      while line.strip():
        self._catagories.append(line.strip())
        line = fd.readline()

    self._imgs_per_batch = 10000
    self._imgs_per_test = 1000

    self._tr_iter = iter(self._training_files)
    self._ts_iter = iter(self._test_files)

    self._curr_file = None
    self._curr_img = 0
    self._img_height = 32
    self._img_width = 32
    self._img_chans = 3
    self._bytes_per_img = self._img_chans * self._img_width * self._img_height + 1
    self._dtype = np.dtype([('catagory', np.uint8), ('image', np.uint8, (self._img_chans, self._img_height, self._img_width))])

  def __iter__(self):
    return self

  def __next__(self):
    if (self._curr_file is None):
      self._curr_img = 0
      if self._training:
        self._curr_file = next(self._tr_iter)
      else:
        self._curr_file = next(self._ts_iter)

    data = None
    with open(self._curr_file, 'r') as fd:
      fd.seek(self._curr_img * self._bytes_per_img)
      d = np.fromfile(fd, self._dtype, 1)
      data = {}
      data['catagory'] = self._catagories[d[0][0]]
      img = d[0][1]
      img = np.swapaxes(img, 0, 2)
      data['image'] = np.swapaxes(img, 0, 1)

    self._curr_img += 1

    max_imgs = self._imgs_per_test
    if (self._training):
      max_imgs = self._imgs_per_batch

    if (self._curr_img > max_imgs):
      self._curr_file = None

    return data


def download(data_dir, data_url):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  filename = data_url.split('/')[-1]
  filepath = os.path.join(data_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(data_dir)
  os.remove(filepath)

if __name__ == '__main__':
  import argparse
  _DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', type=str, default='.',
      help='Directory to download data and extract the tarball')
  parser.add_argument(
      '--data_url', type=str, default=_DATA_URL,
      help='Data URL to download tarball from')

  args = parser.parse_args()
  main(args.data_dir, args.data_url)
