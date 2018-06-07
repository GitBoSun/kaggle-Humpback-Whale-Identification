import sys
import numpy as np
import cv2
import random
import math
import os
import time
import socket
import threading
import Queue
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import scipy.io
import helper
import scipy.io as sio
import scipy.misc as smc
import pandas as pd
from utils import *
BATCH_SIZE=32
max_epoch=225
train_csv='train.csv'
train_path='train'
test_path='test'

class BatchFetcher(threading.Thread):
  def __init__(self, datapath,istrain, isval, repeat):
    super(BatchFetcher, self).__init__()
#    self.queue=Queue.Queue(40)
    if istrain:
      self.queue=Queue.Queue(16)
    else:
      self.queue=Queue.Queue(2)
    self.batch_size=BATCH_SIZE
    self.stopped=False
    self.datadir=datapath
    self.istrain=istrain
    self.repeat=repeat
    self.cur=0
    self.epoch=0
    if self.istrain:
      self.names=self.get_train_names()  
    else:
      self.names=self.get_val_names()
    self.label_dict=self.get_label_dict()
    self.im_dict=self.get_name_dict()
    self.max_step=self.get_max_step()

  def get_train_names(self):
    names=[]
    df=pd.read_csv(train_csv)
    num_im=len(df['Image'])
    num_cat=len(df['Id'].unique())
    images=df['Image']
    labels=df['Id']
    freq=df['Id'].value_counts()
    if not isval:
      for i in range(9000):
        names.append(images[i])
        if labels[i]=='new_whale':
          continue
        for j in range(int(340/freq[labels[i]])):
          names.append(images[i])
    else:
      for i in range(9000, 9850):
        names.append(images[i])
    names=np.random.permutation(names)
    return names

  def get_label_dict(self):
    label_dict={}
    df=pd.read_csv(train_csv)
    cats=df['Id'].unique()
    for i in range(4251):
      label=np.zeros((4251,), np.float32)
      label[i]=1.0
      label_dict[cats[i]]=label
    return label_dict

  def get_name_dict(self):
    df=pd.read_csv(train_csv)
    im_dict={}
    images=df['Image']
    labels=df['Id']
    for i in range(9850):
      im_dict[images[i]]=labels[i]
    return im_dict

  def get_val_names(self):
    pass
    
  def get_max_step(self):
    return int(len(self.names)/self.batch_size)

  def work(self):
    if self.cur+self.batch_size>=len(self.names):
      if self.repeat:
        self.cur=0
        self.epoch+=1
        self.names=np.random.permutation(self.names)
      else:
        self.shutdown()
        return None
    batch_ims=np.zeros((self.batch_size, 300, 500, 1))
    batch_labels=np.zeros((self.batch_size, 4251))
    batch_names=[]
    for i in range(self.batch_size):
      name=self.names[self.cur+i]
      batch_names.append(name)
      image=read_and_preprocess(os.path.join(self.datadir, name), True)
      label=self.label_dict[self.im_dict[name]]
      batch_ims[i]=image
      batch_labels[i]=label
    return [batch_ims, batch_labels, batch_names]

  def run(self):
    if self.cur+self.batch_size>=len(self.names) and self.repeat==False:
      self.shutdown()
    while self.epoch<max_epoch+1 and not self.stopped:
      self.queue.put(self.work())
      print 'push'
      self.cur+=self.batch_size

  def fetch(self):
#    if self.stopped:
#      return None
    return self.queue.get()
  def shutdown(self):
    self.stopped=True
    while not self.queue.empty():
      self.queue.get()
if __name__=='__main__':
  datadir='train'
  fetchworker=BatchFetcher(datadir, True,False, False)
  fetchworker.start()
#  time.sleep(10)
  for i in range(3):
    a,b, c=fetchworker.fetch()
    for j in range(32):
      cv2.imwrite('sample/%s.png'%(c[j]), a[j])
    print a.shape, b.shape, c
  fetchworker.shutdown()
