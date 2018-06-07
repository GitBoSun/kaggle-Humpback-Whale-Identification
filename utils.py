import tensorflow as tf
import numpy as np
import os
import math
import time
import re
import cv2
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.05,
        fill_mode='nearest')

def euclidean_loss(input1_, input2_):
  s = tf.shape(input1_)
  input1 = tf.reshape(input1_, [-1, s[-3],s[-2], s[-1]])
  input2 = tf.reshape(input2_, [-1, s[-3],s[-2], s[-1]])
  return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(input1, input2), 2), 3))
def euclidean_loss2(input1, input2):
  return tf.reduce_mean(tf.pow(tf.subtract(input1, input2),2))

def cross_entroy_loss(pred, label):
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
  class_loss = tf.reduce_mean(loss)
  return class_loss

def linear(input_, output_size, name, bn=False, bn_decay=None, is_training=None,reuse=False):
  msra_coeff = 1.0
  shape = input_.get_shape().as_list()
  fan_in = int(input_.get_shape()[-1])
  stddev = msra_coeff * math.sqrt(2. / float(fan_in))

  with tf.variable_scope(name) as scope:
    if reuse:
      scope.reuse_variables()
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    weight_decay = tf.multiply(tf.nn.l2_loss(matrix), 0.0005)
    tf.add_to_collection('w_losses', weight_decay)
    b = tf.get_variable(
        'b', [output_size, ], initializer=tf.constant_initializer(value=0.))
    outputs = tf.matmul(input_, matrix) + b
  return outputs


def conv2d_bn_relu(input_, output_dim, k_h, k_w, d_h, d_w, name, bn_decay=None, is_training=None, reuse=False,bn=True,relu=True):
  with tf.variable_scope(name) as scope:#, reuse=reuse):
    if reuse:
      scope.reuse_variables()
    msra_coeff = 1.0
    fan_in = k_h * k_w * int(input_.get_shape()[-1])
    stddev = msra_coeff * math.sqrt(2. / float(fan_in))
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#        initializer=tf.truncated_normal_initializer(stddev=stddev))
         initializer= tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
        'b', [output_dim, ],
        initializer=tf.constant_initializer(value=0.0))
#    weight_decay = tf.multiply(tf.nn.l2_loss(w), 0.0005)
#    tf.add_to_collection('w_losses', weight_decay)
    outputs = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME') + b
    if bn:
      outputs = batch_norm_for_conv2d(outputs, is_training,bn_decay=bn_decay, scope='bn')
    if relu:
      outputs=tf.nn.relu(outputs)
    weight_decay = tf.multiply(tf.nn.l2_loss(w), 0.00005)
    tf.add_to_collection('w_losses', weight_decay)
  return outputs

def conv2d(input_,name,  output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, relu=False, reuse=False):
  with tf.variable_scope(name) as scope:#, reuse=reuse):
    if reuse:
      scope.reuse_variables()
    msra_coeff = 1.0
    fan_in = k_h * k_w * int(input_.get_shape()[-1])
    stddev = msra_coeff * math.sqrt(2. / float(fan_in))
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#        initializer=tf.truncated_normal_initializer(stddev=stddev))
         initializer= tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
        'b', [output_dim, ],
        initializer=tf.constant_initializer(value=0.0))
    outputs = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME') + b
    weight_decay = tf.multiply(tf.nn.l2_loss(w), 0.00005)
    tf.add_to_collection('w_losses', weight_decay)
    if relu:
      outputs=tf.nn.relu(outputs)
  return outputs

def max_pool(inputs, name):
  with tf.variable_scope(name):
    output=tf.nn.max_pool(inputs, [1,2,2,1], [1,2,2,1], padding='SAME')
  return output

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  # return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)
  with tf.variable_scope(scope):
    # this block looks like it has 3 inputs on the graph unless we do this
    inputs = tf.identity(inputs)
    channels = inputs.get_shape()[3]
    offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
    scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
    variance_epsilon = 1e-5
    normalized = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def dropout(inputs,is_training,scope,keep_prob=0.5,noise_shape=None):
  """ Dropout layer.
  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints
  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs

def load_snapshot(saver, session, path):
  if not os.path.exists(path):
    print '%s not exist' % path
    return 0
  assert os.path.exists(path), ('%s not exist' % path)
  ckpt = tf.train.get_checkpoint_state(path)
  if ckpt is not None:
    model_checkpoint_path = os.path.join(path, os.path.basename(ckpt.model_checkpoint_path))
    print("loading model " + model_checkpoint_path + "...")
    saver.restore(session, model_checkpoint_path)
    num_iter = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))
    print("done.")
    return num_iter
  else:
    print '%s not exist' % path
    return 0

#data agumentation
def read_and_preprocess(path, is_train):
  """
  color->gray, [512, 256], histgram equ, 
  """
  im=cv2.imread(path, 0)
  # im = cv2.equalizeHist(im)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  im = clahe.apply(im)
  # im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  h, w=im.shape
  # print h, w
  #reshape to (512, 256)
  # if 2*h>w:
  #   im=cv2.copyMakeBorder(im,10,10,(2*h+20-w)/2,(2*h+20-w)/2,cv2.BORDER_REPLICATE)
  # if 2*h<w:
  #   im=cv2.copyMakeBorder(im,(w+20-2*h)/2,(w+20-2*h)/2,10, 10, cv2.BORDER_REPLICATE)
  # im=cv2.resize(im, (512, 256), cv2.INTER_CUBIC)
  im=cv2.resize(im, (500, 300), cv2.INTER_CUBIC)
  im=np.expand_dims(im, -1)
  if is_train:
    im=datagen.random_transform(im)
  # im = np.expand_dims(cv2.equalizeHist(im[:,:,0]), -1)
  return im




