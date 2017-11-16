#the baseline alexnet for train flower17 dataset
#architecture is from tflearn
import numpy as np
import tensorflow as tf

import os
import time

def shuffle(*arrs):
	""" shuffle.
	Shuffle given arrays at unison, along first axis.
	Arguments:
	*arrs: Each array to shuffle at unison.
	Returns:
	Tuple of shuffled arrays.
	"""
	arrs = list(arrs)
	for i, arr in enumerate(arrs):
		assert len(arrs[0]) == len(arrs[i])
		arrs[i] = np.array(arr)
	p = np.random.permutation(len(arrs[0]))	
	return tuple(arr[p] for arr in arrs)

#load the data
X = np.load('FlowerX.npy')
Y = np.load('FlowerY.npy')

XTrain = X[0:1000]
YTrain = Y[0:1000]
#shuffle the data
XTrain, YTrain = shuffle(XTrain, YTrain)
print('shuffle done')

SizeBatch = 64
BatchIndex = 0
SizeTrainData = YTrain.shape[0]

def train_next_batch(SizeBatch):
	global XTrain
	global YTrain
	global BatchIndex
	global SizeTrainData
	if(BatchIndex + SizeBatch > SizeTrainData):#shuffle the data
		XTrain, YTrain = shuffle(XTrain, YTrain)
		Start = 0
		End = SizeBatch
		BatchIndex = SizeBatch
	else:
		Start = BatchIndex
		End = BatchIndex + SizeBatch
		BatchIndex = BatchIndex + SizeBatch
	BatchData = XTrain[Start:End]
	BatchLabels = YTrain[Start:End]
	return BatchData, BatchLabels

#define the placeholder for the input images and labels
X = tf.placeholder(tf.float32, [SizeBatch, 227, 227, 3])
Y_ = tf.placeholder(tf.float32, [SizeBatch, 10])

#definition of weights in 1st conv layer
WConv1 = tf.Variable(tf.truncated_normal([11, 11, 3, 64], 
                                         stddev = 0.1))
BConv1 = tf.Variable(tf.zeros([64]))
#1st conv layer
LayerConv1 = tf.nn.relu(tf.nn.conv2d(X, WConv1, strides=[1, 4, 4, 1], padding = 'SAME') + BConv1)
#1st pool layer
LayerPool1 = tf.nn.max_pool(LayerConv1, 
                            ksize=[1, 3, 3, 1], 
			    strides=[1, 2, 2, 1], 
			    padding='VALID')
#1st lrn
LayerLRN1 = tf.nn.local_response_normalization(LayerPool1,
                                               alpha=1e-4,
                                               beta=0.75,
                                               depth_radius=2, 
					       bias=2.0)
#definition of weights in 2nd conv layer
WConv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 192], 
                                         stddev = 0.1))
BConv2 = tf.Variable(tf.zeros([192]))
#2nd conv layer
LayerConv2 = tf.nn.relu(tf.nn.conv2d(LayerLRN1, WConv2, strides=[1, 1, 1, 1], padding = 'SAME') + BConv2)
#2nd pool layer
LayerPool2 = tf.nn.max_pool(LayerConv2,
                            ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID')
#2nd lrn layer
LayerLRN2 = tf.nn.local_response_normalization(LayerPool2,
                                               alpha=1e-4,
                                               beta=0.75,
                                               depth_radius=2,
                                               bias=2.0)
