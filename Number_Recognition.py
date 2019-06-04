#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:19:51 2019

@author: madhavan5
"""

import tensorflow as tf
import pandas as pd
import numpy as np

def rand_batch(no_batches, dependent, independent):

    idx = np.arange(0 , len(dependent))
    np.random.shuffle(idx)
    idx = idx[:no_batches]
    dependent_shuffled = [dependent[ i] for i in idx]
    independent_shuffled = [independent[ i] for i in idx]
    
    independent_shuffled = np.asarray(independent_shuffled)
    zero = np.zeros((independent_shuffled.size, (independent_shuffled.max()+1)))
    zero[np.arange(independent_shuffled.size), independent_shuffled] = 1

    return np.asarray(dependent_shuffled), zero


learning_rate = 0.001
epochs = 100
batch_size = 400

train = pd.read_csv("train.csv")
train = train.as_matrix()

test = pd.read_csv("test.csv")
test = test.as_matrix()

Dependent = tf.placeholder(tf.float32,[None,784],"input")
Independent = tf.placeholder(tf.float32,[None,10],"output")

W1 = tf.Variable(tf.random_normal([784,300], stddev=0.03), name='W1')
B1 = tf.Variable(tf.random_normal([300]), name='B1')

W2 = tf.Variable(tf.random_normal([300,10], stddev=0.03), name='W2')
B2 = tf.Variable(tf.random_normal([10]), name='B2')


hidden_output = tf.nn.relu(tf.add(tf.matmul(Dependent,W1),B1))

output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_output,W2),B2))

clipped_output = tf.clip_by_value(output_layer,1e-10,0.9999999)


cross_entropy = -tf.reduce_mean(tf.reduce_sum(Independent*tf.log(clipped_output) + (1-Independent)*tf.log(1 - clipped_output), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

initializer = tf.global_variables_initializer()

prediction = tf.equal(tf.argmax(Independent,1), tf.argmax(output_layer,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
guess = tf.argmax(output_layer,1)



with tf.Session() as sess:
    sess.run(initializer)

    total_no_of_batches = int(len(train)/batch_size)
    

    for epoch in range(epochs):

        avg_cost=0

        for batch in range(total_no_of_batches):

            batch_x, batch_y = rand_batch(batch_size, train[0:40001,1:], train[0:40001,0])
            _, cost = sess.run([optimizer, cross_entropy], feed_dict={Dependent: batch_x, Independent: batch_y})
            
            avg_cost += cost/total_no_of_batches
        print("Epoch:",(epoch+1),"cost = ",avg_cost)
    batch_x1, batch_y1 = rand_batch(batch_size, train[40001:,1:], train[40001:,0])
    print("Accuracy :",sess.run(accuracy, feed_dict={Dependent: batch_x1, Independent: batch_y1}))
    answer = sess.run(guess,feed_dict={Dependent: test[:,:]})
    
num = np.arange(1,28001,dtype=int)
result = np.vstack((num,answer)).T
np.savetxt("First.csv", result, delimiter=",")