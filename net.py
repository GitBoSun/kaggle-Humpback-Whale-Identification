import tensorflow as tf
import numpy as np
from utils import *
import os
from inceptionv4 import *

class Inception_whale(object):
  def __init__(self):
    self.batch_size=32
    self.s_in_w=500
    self.s_in_h=300
    self.num_class=2451
    self.aux_logits=True

  def set_placeholder(self):
    self.im_in=tf.placeholder(tf.float32,(self.batch_size,self.s_in_h, self.s_in_w,1),name='im_in')
    self.gt_label=tf.placeholder(tf.float32, (self.batch_size, self.num_calss), name='class_label')
    self.is_training = tf.placeholder(tf.bool, shape=())
  def set_feed(self, image, label, step, is_training=True):
    self.feed_dict={
      self.im_in:image,
      self.gt_label:label,
      self.global_step:step,
      # self.epoch=epoch,
      self.is_training:is_training
    }
  def get_bn_decay(self, batch):
    bn_momentum = tf.train.exponential_decay(0.5, batch * 20, 200000, 0.5,staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

  def get_lr(self, batch):
    lr=tf.train.exponential_decay(0.0005, batch, 20000, 0.5, staircase=True)
    return lr

  def build_network(self):
    self.out_label, self.end_points = inception_v4(self.im_in, num_classes=self.num_class, is_training=self.is_training,create_aux_logits=self.aux_logits )
    if self.aux_logits:
      self.aux_label=self.end_points['AuxLogits']
    print 'Inception input:', self.im_in
    print 'Inception output:', self.out_label
   
  def set_loss(self):
    self.loss=tf.nn.softmax_cross_entropy_with_logits(self.out_label, self.gt_label)
    if self.aux_logits:
      self.loss+=tf.nn.softmax_cross_entropy_with_logits(self.aux_label, self.gt_label)
    tf.summary.scalar('loss', self.loss)

  def run_optim(self,sess):
    return sess.run(self.optim, feed_dict=self.feed_dict)
  def run_loss(self, sess):
    return sess.run(self.loss,feed_dict=self.feed_dict)
  def run_result(self,sess):
    return sess.run(self.out_label,feed_dict=self.feed_dict)
  def run_sum(self,sess):
    return sess.run(self.merged, feed_dict=self.feed_dict)
  def save_model(self, sess, folder, it):
    self.saver.save(sess, os.path.join(folder, "model"+str(it)), global_step=it)
  def restore_model(self, sess, folder):
    print 'restore all variables', folder
    return load_snapshot(self.saver, sess, folder)
  def print_loss_acc(self, sess):
    self.acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.out_label,1), tf.argmax(self.gt_label,1)), tf.float32))
    output=sess.run([self.loss,self.acc] , feed_dict=self.feed_dict)
    print ('[Loss: %.4f] [acc: %.4f]' % (output[0], output[1]))
  def run_step(self,sess):
    step_=sess.run(self.global_step)
    return step_
  def step_assign(self,sess, i):
    step_p=self.global_step.assign(i)
    sess.run(step_p)

  def build_model(self):
    self.global_step=tf.Variable(0, trainable=False)
    self.learning_rate=self.get_lr(self.global_step)
    self.bn_decay=self.get_bn_decay(self.global_step)
    self.set_placeholder()
    self.build_network()
    self.set_loss()
    self.merged=tf.summary.merge_all()
    self.t_vars=tf.trainable_variables() 
    self.optim=tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.loss, var_list=self.t_vars)
    #self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.t_vars)
    self.saver = tf.train.Saver(self.t_vars, max_to_keep=3)
