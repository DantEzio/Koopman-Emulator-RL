# -*- coding: utf-8 -*-
"""
@author: Wenchong
"""

import tensorflow as tf
import numpy as np


def weight_variable(shape, var_name, distribution='tn', scale=0.1):
    if distribution == 'tn':
        initial = tf.compat.v1.truncated_normal(shape, stddev=scale, dtype=tf.float64)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.compat.v1.random_normal(shape, mean=0, stddev=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
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






class DLEDMD():
    def __init__(self,params,X_train,Y_train,X_train_F):
        super(DLEDMD, self).__init__()
        tf.compat.v1.disable_eager_execution()
        
        self.x = tf.compat.v1.placeholder(tf.float64, [None, X_train.shape[1]],name='xin')
        self.f = tf.compat.v1.placeholder(tf.float64, [None, X_train_F.shape[1]],name='fin')
        self.y = tf.compat.v1.placeholder(tf.float64, [None, Y_train.shape[1]],name='yin')

        self.params=params
        self.weights = dict()
        self.biases = dict()
        self.enlayers()
        self.delayers()
        self.Klayer()
        
    def enlayers(self):
        widths=self.params['enlayers']#width be like [[2,3],[3,4]]
        for i in np.arange(len(widths)):
            self.weights['We%d' % (i)] = weight_variable([widths[i][0], widths[i][1]], var_name='We%d' % (i))
            self.biases['be%d' % (i)] = bias_variable([widths[i][1], ], var_name='be%d' % (i))
    
    def delayers(self):
        widths=self.params['delayers']#width be like [[2,3],[3,4]]
        for i in np.arange(len(widths)):
            self.weights['Wd%d' % (i)] = weight_variable([widths[i][0], widths[i][1]], var_name='Wd%d' % (i))
            self.biases['bd%d' % (i)] = bias_variable([widths[i][1], ], var_name='bd%d' % (i))
    
    def Klayer(self):
        widths=self.params['K']#width be like [3,4]
        self.weights['K'] = weight_variable([widths[0], widths[1]], var_name='K')
    
    '''
    Apply net#################################################################################################################
    '''
    
    def layer_apply(self,prev_layer):
        #prev_layer=tf.concat([x_encoder,f_encoder],axis=1)
        with tf.compat.v1.variable_scope('enlayers',reuse=tf.compat.v1.AUTO_REUSE):
            widths=self.params['enlayers']
            for i in np.arange(len(widths)):
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, self.weights['We%d' % (i)]) + self.biases['be%d' % (i)])
        
        with tf.compat.v1.variable_scope('Klayers',reuse=tf.compat.v1.AUTO_REUSE):
            #widths=self.params['K']
            prev_layer = tf.matmul(prev_layer, self.weights['K'])
                            
        with tf.compat.v1.variable_scope('delayers',reuse=tf.compat.v1.AUTO_REUSE):
            widths=self.params['delayers']
            for i in np.arange(len(widths)-1):
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, self.weights['Wd%d' % (i)]) + self.biases['bd%d' % (i)])
            
        return tf.nn.relu(tf.matmul(prev_layer, self.weights['Wd%d' % (len(widths)-1)]) + self.biases['bd%d' % (len(widths)-1)])
        
    
    def forward_net(self,x,f):
        xf=tf.concat((x,f),axis=1)
        y=self.layer_apply(xf)
        return y 
    
    def Loss(self,x,f,y):
        def loss1(x,f,y):
            with tf.compat.v1.variable_scope('loss1',reuse=tf.compat.v1.AUTO_REUSE):
                return tf.reduce_mean(tf.square(self.forward_net(x,f)-y))

        def loss2(x,f,y):
            xt=tf.reshape(x[0],(1,self.params['input_layer']))
            with tf.compat.v1.variable_scope('loss2',reuse=tf.compat.v1.AUTO_REUSE):
                N,m=y.shape
                err_bar=[]
                for i in np.arange(self.params['step']):
                    y_bar=self.forward_net(xt,tf.reshape(f[i],(1,self.params['f_layer'])))
                    err_bar.append(y_bar-y[i])
                    xt=y_bar
                return tf.reduce_mean(tf.square(err_bar))
        
        
        #loss2=self.loss2(x,f,y)
        loss=loss1(x,f,y)
        return loss
    
    def opt(self,loss):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.params['lr'])
        return self.optimizer.minimize(loss)
