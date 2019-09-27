from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


ROW_SIZE = 112
COL_SIZE = 112
FRAME_NUM = 25
#NUM_CLASSES = 10
NUM_CLASSES = 21

def read_LRWDataset(filename_queue):

  class LRWDataset(object):
    pass
  result = LRWDataset()
  result.height = ROW_SIZE
  result.width = COL_SIZE
  result.depth = 3
  image_bytes = result.height * result.width * result.depth * FRAME_NUM
  
  # Every record consists of a label followed by the image, with a fixed number of bytes for each.
  record_bytes = image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  depth_major = tf.reshape(record_bytes, [FRAME_NUM, result.depth, result.height, result.width])  
  
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [0, 2, 3, 1])
  return result

def inputs():
  filename = ['test_batch.bin']

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filename)

  # Read examples from files in the filename queue.
  read_input = read_LRWDataset(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.int32)

  reshaped_image.set_shape([FRAME_NUM, ROW_SIZE, COL_SIZE, 3])
  transformed_images = []
  for i in range(FRAME_NUM):
    transformed_images.append(tf.expand_dims(tf.image.per_image_standardization(reshaped_image[i, :, :, :]), 0))
  reshaped_image_norm = tf.concat(transformed_images,0)

  image = tf.train.batch([reshaped_image_norm],1)
  return image                                 
