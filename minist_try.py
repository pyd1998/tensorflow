#!/usr/bin/env python
# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
minist=input_data.read_data_sets("./MINIST_data/",one_hot=True)
print "training data size: ",minist.train.num_examples
print "validating data size: ",minist.validation.num_examples
print "testing data size: ",minist.test.num_examples
print "Example training data: ",minist.train.images[0]
print "example training label: ",minist.train.labels[0]
