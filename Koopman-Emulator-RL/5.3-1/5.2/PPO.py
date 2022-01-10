# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf


class PPO:
    def __init__(self, env, ep, batch, num_rain,t, raindata='test'):
        self.t = t
        self.ep = ep
        
        self.action_size=8

        #self.env = gym.make('Pendulum-v0')
        self.env = env
        self.batch = env.T

        self.bound_high = np.array([1 for _ in range(self.action_size)])
        self.bound_low = np.array([0 for _ in range(self.action_size)])

        self.gamma = 0.9
        self.LR = 0.001
        self.UPDATE_STEPS = 10
        self.kl_target = 0.01
        self.lam = 0.5
        self.epsilon = 0.2
        self.num_rain=num_rain

        self.sess = tf.compat.v1.Session()
        
        #tf.reset_default_graph()
        self.build_model()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.raindata=raindata
        if raindata=='test':
            self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        else:
            self.testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#读取真实降雨数据
        
        self.rainnum,m=self.rainData.shape
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
        
        

    def _build_critic(self):
        with tf.compat.v1.variable_scope('critic_ppo',reuse=tf.compat.v1.AUTO_REUSE):
            #x = tf.layers.dense(self.states, 100, tf.nn.relu,kernel_initializer=tf.zeros_initializer(),bias_initializer=tf.zeros_initializer())
            x = tf.compat.v1.layers.dense(self.states, 100, tf.nn.relu,kernel_initializer=tf.zeros_initializer(),bias_initializer=tf.zeros_initializer())

            self.v = tf.compat.v1.layers.dense(x, 1,kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
            self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
            x = tf.compat.v1.layers.dense(self.states, 100, tf.nn.relu, 
                                trainable=trainable,kernel_initializer=tf.zeros_initializer(), 
                                bias_initializer=tf.zeros_initializer(),
                                reuse=tf.compat.v1.AUTO_REUSE)

            mu = (self.bound_high-self.bound_low) * tf.compat.v1.layers.dense(x, self.action_size, tf.nn.tanh, trainable=trainable,
                                                                    kernel_initializer=tf.zeros_initializer(), 
                                                                    bias_initializer=tf.zeros_initializer())-self.bound_low
            sigma = tf.compat.v1.layers.dense(x, self.action_size, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)

        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params

    def build_model(self):
        """build model with ppo loss.
        """
        # inputs
        self.states = tf.compat.v1.placeholder(tf.float32, [None, 5], 'states')
        self.action = tf.compat.v1.placeholder(tf.float32, [None, 8], 'action')#8 pumps are controled
        self.adv = tf.compat.v1.placeholder(tf.float32, [None, 1], 'advantage')
        self.dr = tf.compat.v1.placeholder(tf.float32, [None, 1], 'discounted_r')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor_ppo', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_ppo_actor', trainable=False)

        # define ppo loss
        with tf.compat.v1.variable_scope('loss',reuse=tf.compat.v1.AUTO_REUSE):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor loss
            with tf.compat.v1.variable_scope('surrogate'):
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
                surr = ratio * self.adv

            if self.t == 'ppo1':
                self.tflam = tf.compat.v1.placeholder(tf.float32, None, 'lambda')
                kl = tf.compat.v1.distributions.kl_divergence(old_nd, nd)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else: 
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))

        # define Optimizer
        with tf.compat.v1.variable_scope('optimize',reuse=tf.compat.v1.AUTO_REUSE):
            self.ctrain_op = tf.compat.v1.train.AdamOptimizer(self.LR).minimize(self.closs)
            self.atrain_op = tf.compat.v1.train.AdamOptimizer(self.LR).minimize(self.aloss)

        with tf.compat.v1.variable_scope('sample_action',reuse=tf.compat.v1.AUTO_REUSE):
            self.sample_op = tf.squeeze(nd.sample(self.action_size), axis=1)

        # update old actor
        with tf.compat.v1.variable_scope('update_old_actor',reuse=tf.compat.v1.AUTO_REUSE):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        #tf.compat.v1.summary.FileWriter(self.log, self.sess.graph)


    def choose_action(self, state):
        
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]
        for i in range(action.shape[0]):
            if action[i]<0.5:
                action[i]=0
            else:
                action[i]=1
        #return np.clip(action, -self.bound, self.bound)
        return np.clip(action, self.bound_low, self.bound_high)

    def get_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.states: state})

    def discount_reward(self, states, rewards, next_observation):
        n=len(states[0])
        s = np.vstack([states, next_observation.reshape(-1, n)])
        q_values = self.get_value(s).flatten()

        targets = rewards + self.gamma * q_values[1:]
        targets = targets.reshape(-1, 1)

        return targets

    def update(self, states, action, dr):
        self.sess.run(self.update_old_actor)

        adv = self.sess.run(self.advantage,
                            {self.states: states,
                             self.dr: dr})

        # update actor
        if self.t == 'ppo1':
            # run ppo1 loss
            for _ in range(self.UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.states: states,
                     self.action: action,
                     self.adv: adv,
                     self.tflam: self.lam})

            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            # run ppo2 loss
            for _ in range(self.UPDATE_STEPS):
                self.sess.run(self.atrain_op,
                              {self.states: states,
                               self.action: action,
                               self.adv: adv})

        # update critic
        for _ in range(self.UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})

    def train(self,save):
        for j in range(self.ep):
            for i in range(self.num_rain):
                print('steps: ',j,' rain:',i)
                observation,_ = self.env.reset(self.rainData[i])
                states, actions, rewards = [], [], []
                
                while True:
                    a = self.choose_action(observation)
                    #print(a)
                    next_observation, reward, done, _, _ = self.env.step(a,self.rainData[i])
                    states.append(observation)
                    actions.append(a)
                    rewards.append(reward)
                    observation = next_observation
                    
                    if done:
                        break
                    
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            d_reward = self.discount_reward(states, rewards, next_observation)
            self.update(states, actions, d_reward)
        #保存模型
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
            print('test'+str(i))
            observation,flooding = self.env.reset(self.testRainData[i])
            
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            hc_name='./'+self.t+'_test_result/HC/HC'+str(i)
            self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            hc_flooding = self.env.reset_HC(hc_name)
            flooding_log,hc_flooding_log=[flooding],[hc_flooding]
            
            states, actions, rewards = [], [], []
            
            while True:
                a = self.choose_action(observation)
                next_observation, reward, done, _, flooding = self.env.step(a,self.testRainData[i])
                #对比HC,也记录HC每一步的flooding
                _, hc_flooding = self.env.step_HC(hc_name)
                states.append(observation)
                actions.append(a)
                flooding_log.append(flooding)
                hc_flooding_log.append(hc_flooding)
                rewards.append(reward)
                observation = next_observation
                    
                if done:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.discount_reward(states, rewards, next_observation)

                    states, actions, rewards = [], [], []
                    break
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            hc_flooding_logs.append(hc_flooding_log)
            
            #save RLC .inp and .rpt
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
            #self.env.copy_result(sout,sin)
            #保存所有降雨的flooding过程线
            df = pd.DataFrame(np.array(flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
            df = pd.DataFrame(np.array(hc_flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
        

    def save_history(self, history, name):
        #name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


if __name__ == '__main__':
    '''
    model1 = PPO(1000, 32, 'ppo1')
    history = model1.train()
    model1.save_history(history, 'ppo1.csv')
    '''
    model2 = PPO(1000, 32, 'ppo2')
    history = model2.train()
    model2.save_history(history, 'ppo2.csv')