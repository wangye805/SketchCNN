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


