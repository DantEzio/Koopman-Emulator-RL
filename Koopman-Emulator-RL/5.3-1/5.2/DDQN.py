# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import pandas as pd

np.random.seed(1)
tf.compat.v1.set_random_seed(1)


class DDQN:
    def __init__(self,step,batch_size,num_rain,env,t,raindata):
        
        #action与实际操作的对应表
        self.action_table=pd.read_excel('./action_table_of_DQN.xlsx').values[:,1:]
        ACTION_SPACE = self.action_table.shape[0]
        print('table shape:',self.action_table.shape)
        
        
        n_features=5
        memory_size= 150000
        e_greedy=0.01
        reward_decay=0.01
        learning_rate=0.001
        e_greedy_increment=0.1
        replace_target_iter=10
        output_graph=False
        
        self.t=t
        self.env=env
        
        self.n_actions = ACTION_SPACE
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = self.env.T
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.num_rain=num_rain

        self.traing_step=step
        
        if self.t=='ddqn':
            self.dueling = True      # decide to use dueling DQN or not
        else:
            self.dueling = False     # decide to use dueling DQN or not

        self.learn_step_counter = 0
        #self.memory = np.zeros((self.memory_size, n_features*2+2))
        self.memory=[]
        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]
        
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.raindata=raindata
        if raindata=='test':
            self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        else:
            self.testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#读取真实降雨数据
        
        self.rainnum,m=self.rainData.shape     
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
        
        
        
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.compat.v1.variable_scope('l1',reuse=tf.compat.v1.AUTO_REUSE):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.compat.v1.variable_scope('Value',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.compat.v1.variable_scope('Advantage',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.compat.v1.variable_scope('Q',reuse=tf.compat.v1.AUTO_REUSE):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.compat.v1.variable_scope('Q',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.compat.v1.variable_scope('eval_net',reuse=tf.compat.v1.AUTO_REUSE):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.4), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.compat.v1.variable_scope('loss',reuse=tf.compat.v1.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train',reuse=tf.compat.v1.AUTO_REUSE):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.compat.v1.variable_scope('target_net',reuse=tf.compat.v1.AUTO_REUSE):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r], s_))
        self.memory.append(transition)
        self.memory_counter += 1
        

    def choose_action(self, observation):
        #observation = observation[np.newaxis, :]
        pa=np.random.uniform()
        if pa < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
            action = np.argmax(actions_value)  
        else:
            action = np.random.randint(0, self.n_actions)  
        return action
    
    def choose_action_test(self, observation):
        #observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
        action = np.argmax(actions_value)
        return action

    def learn(self,total_step):
        self.sess.run(self.replace_target_op)
        sample_index = np.random.choice(total_step, size=self.batch_size)
        batch_memory=[]
        for i in sample_index:
            batch_memory.append(list(self.memory[int(i)]))
        batch_memory=np.array(batch_memory)
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:,-self.n_features:]}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:,:self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index] = np.reshape(reward + self.gamma * np.max(q_next, axis=1),(self.batch_size,1))
        
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
    def train(self,save):
        
        for j in range(self.traing_step):
            total_steps = 0
            for i in range(self.num_rain):
                print('training steps:',j)
                print('sampling number:',i)
                observation, _ = self.env.reset(self.rainData[i])
                while True:
                    tema = self.choose_action(observation)
                    action = self.action_table[tema,:].tolist()
                    observation_, reward, done, info, _ = self.env.step(action,self.rainData[i])
                    self.store_transition(observation, tema, reward, observation_)
                    observation = observation_
                    total_steps += 1
                
                    if done:
                        break
                
                if total_steps-self.memory_size > 15000:
                    break
            
            self.learn(total_steps)  
            if save:
                saver=tf.compat.v1.train.Saver()
                sp=saver.save(self.sess,'./'+self.t+'_test_result/model/'+self.t+'_model.ckpt')
                print("model saved:",sp)
    
    def load_model(self):
        saver=tf.compat.v1.train.Saver()
        saver.restore(self.sess,'./'+self.t+'_test_result/model/'+self.t+'_model.ckpt')
            
    
    def test(self,test_num):
        flooding_logs,hc_flooding_logs=[],[]
        for i in range(test_num):
            print('test',i)
            
            observation, flooding = self.env.reset(self.testRainData[i])
            hc_name='./'+self.t+'_test_result/HC/HC'+str(i)
            self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            hc_flooding = self.env.reset_HC(hc_name)
            
            flooding_log,hc_flooding_log=[flooding],[hc_flooding]

            while True:
                tema = self.choose_action_test(observation)
                action = self.action_table[tema,:].tolist()
                observation_, reward, done, info, flooding = self.env.step(action,self.testRainData[i])
                _, hc_flooding = self.env.step_HC(hc_name)
                
                flooding_log.append(flooding)
                hc_flooding_log.append(hc_flooding)
                observation = observation_
                if done:
                    break
            
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            hc_flooding_logs.append(hc_flooding_log)
            
            if self.raindata=='test':
                k=0
            else:
                k=4
            sout='./'+self.t+'_test_result/'+str(i+k)+'.rpt'
            sin=self.env.staf+'.rpt'
            self.env.copy_result(sout,sin)
            sout='./'+self.t+'_test_result/'+str(i+k)+'.inp'
            sin=self.env.staf+'.inp'
            self.env.copy_result(sout,sin)
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
        df = pd.DataFrame(np.array(hc_flooding_logs).T)
        df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
    
    def save_history(self, history, name):
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
    
    
    