# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def weight_variable(shape, var_name, distribution='tn', scale=0.1):
    if distribution == 'tn':
        initial = tf.compat.v1.truncated_normal(shape, stddev=scale, dtype=tf.float64)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'dl':
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'he':
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.compat.v1.random_normal(shape, mean=0, stddev=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name, distribution=''):
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    else:
        initial = tf.compat.v1.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)


class MLP():
    def __init__(self,params,X_train,Y_train,X_train_F):
        super(MLP, self).__init__()
        tf.compat.v1.disable_eager_execution()
        
        self.x = tf.compat.v1.placeholder(tf.float64, [None, X_train.shape[1]],name='xin')
        self.f = tf.compat.v1.placeholder(tf.float64, [None, X_train_F.shape[1]],name='fin')
        self.y = tf.compat.v1.placeholder(tf.float64, [None, Y_train.shape[1]],name='yin')

        self.params=params
        self.weights = dict()
        self.biases = dict()
        self.layers()
        
    def layers(self):
        widths=self.params['layers']#width be like [[2,3],[3,4]]
        for i in np.arange(len(widths)):
            self.weights['W%d' % (i)] = weight_variable([widths[i][0], widths[i][1]], var_name='W%d' % (i))
            self.biases['b%d' % (i)] = bias_variable([widths[i][1], ], var_name='b%d' % (i))

    def layer_apply(self,prev_layer):
        #prev_layer=tf.concat([x_encoder,f_encoder],axis=1)
        with tf.compat.v1.variable_scope('layers',reuse=tf.compat.v1.AUTO_REUSE):
            widths=self.params['layers']
            for i in np.arange(len(widths)):
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, self.weights['W%d' % (i)]) + self.biases['b%d' % (i)])
            
        return prev_layer
        
    
    def forward_net(self,x,f):
        xf=tf.concat((x,f),axis=1)
        y=self.layer_apply(xf)
        return y 
    
    def Loss(self,x,f,y):
        with tf.compat.v1.variable_scope('loss1',reuse=tf.compat.v1.AUTO_REUSE):
            loss=tf.reduce_mean(tf.square(self.forward_net(x,f)-y))
            return loss
    
    def opt(self,loss):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.params['lr'])
        return self.optimizer.minimize(loss)
