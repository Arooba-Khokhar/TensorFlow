from __future__ import print_function

import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import sklearn
from sklearn import datasets


LOGDIR = "C:/Users/stech/Desktop/2nd semester/DDA/lab 5"


def func(var):
    with tf.name_scope('summaries'):
            
        mean = var
        tf.summary.scalar('mean',mean)
        tf.summary.histogram('histogram',mean)

olive = datasets.fetch_olivetti_faces()

data = olive.data
print(len(data[0]))

label = np.zeros((400, 40), dtype=np.int)

for i in range(400):
    label[i][olive.target[i]] = 1

labels = label

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.10, random_state=42)

# Parameters
learning_rate = 0.02
epochs = 150    # Any arbitrary number
display_step = 1

increment = 10

# tf Graph Input
Parameters = tf.placeholder(tf.float32, [None,4096], name='Parameters') 
Classes = tf.placeholder(tf.float32, [None,40], name='Classes')
# Set model weights
Weight = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal([4096,40],mean=0.0,stddev=0.05))
Bias = tf.Variable(tf.zeros([40]))

# Construct model
prediction = tf.nn.softmax(tf.matmul(Parameters, Weight) + Bias) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Classes*tf.log(prediction), reduction_indices=1))

# Gradient Descent
GD_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Classes, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Initializing the variables
init = tf.global_variables_initializer()
    
tf.summary.scalar("loss", cost)

'''
# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
'''

'''
# Record that distribution into a histogram summary
summaries = tf.summary.histogram("normal/moving_mean", mean_moving_normal)
'''

tf.summary.histogram('accuracy',accuracy)
tf.summary.histogram("loss", cost)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()  


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    summary_writer = tf.summary.FileWriter(LOGDIR+'starter', graph=tf.get_default_graph())

    total_batch = 40
    
    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.0
        
        batch_size = 0      
        
        # Loop over all batches
        for i in range(total_batch):
            Parameter_batch = data_train[batch_size:batch_size+increment]
            #print((Parameter_batch).shape)
            Classes_batch = labels_train[batch_size:batch_size+increment]
            #print(Classes_batch.shape)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([GD_optimizer, cost, merged_summary_op],
                                     feed_dict={Parameters: Parameter_batch,
                                                Classes: Classes_batch})
#            _, c= sess.run([GD_optimizer, cost],
#                                     feed_dict={Parameters: Parameter_batch,
#                                                Classes: Classes_batch})
                        
            # Compute average loss
            avg_cost += c / total_batch
            
            #summary_writer.add_summary(summary, epoch * total_batch + i)
            
            batch_size += increment
    
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("\n\nOptimization Finished!")
    
    print("\n\nNow Testing on Test Set")
    
    print("\n\nAccuracy on Test Set is : ", accuracy.eval({Parameters: data_test, Classes: labels_test}))
    
    print("\n\n")
