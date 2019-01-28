#!/usr/bin/env python
# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#minist=input_data.read_data_sets("./MINIST_data/",one_hot=True)
#print "training data size: ",minist.train.num_examples
#print "validating data size: ",minist.validation.num_examples
#print "testing data size: ",minist.test.num_examples
#print "Example training data: ",minist.train.images[0]
#print "example training label: ",minist.train.labels[0]

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAIN_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def interface(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)+avg_class.average(biases2))

def train(minist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    y=interface(x,None,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)
    variables_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op=variables_averages.apply(tf.trainable_variables())
    average_y=interface(x,variables_averages,weights1,biases1,weights2,biases2)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,minist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed={x:minist.validation.images,y_:minist.validation.labels}
        test_feed={x:minist.test.images,y_:minist.test.labels}
        for i in range(TRAIN_STEPS):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print "after ",i," steps,validate accuracy is ",validate_acc
            xs,ys=minist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print "after ",TRAIN_STEPS," steps,test accuracy is ",test_acc

def main(argv=None):
    minist=input_data.read_data_sets("./MINIST_data/",one_hot=True)
    train(minist)
if __name__=='__main__':
    tf.app.run()
