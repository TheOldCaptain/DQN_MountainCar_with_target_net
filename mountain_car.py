'''
以汽车的动力是无法使小车爬上坡的，
'''

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import sys
import time

MAX_DEQUE_SIZE = 500  # 样本队列的最大长度
BATCH_SIZE = 32     # minibatch的大小
GAMMA = 0.9
EPSILON = 0.5

class car_DQN:

    def __init__(self, env):
        self.sample_deque = deque()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.epsilon = EPSILON

        self.create_network()
        self.network_loss_function()


        # 初始化会话
        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.reload_network()

    def create_network(self):
        '''
        创建神经网络的框架
        layer1： layer1_out =W1*x_input + b1
                a_layer1 = relu(layer_out)
        layer2:  layer2_out = W2*a_layer1 + b2
        :return:
        '''
        # 输入层 Input layer
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        # 隐藏层 hidden layer1
        self.W1 = self.weight_variable([self.state_dim, 20])
        self.b1 = self.bias_variable([20])
        layer1_out = tf.nn.relu(tf.matmul(self.state_input, self.W1) + self.b1)
        # 输出层 output layer
        self.W2 = self.weight_variable([20, self.action_dim])
        self.b2 = self.bias_variable([self.action_dim])
        self.network_out = tf.nn.relu(tf.matmul(layer1_out, self.W2) + self.b2)

        # TargetNetwork 的创建
        # 输入层 Input layer
        self.target_state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        # 隐藏层 hidden layer
        self.target_W1 = self.weight_variable([self.state_dim, 20])
        self.target_b1 = self.bias_variable([20])
        target_layer1_out = tf.nn.relu(tf.matmul(self.target_state_input, self.target_W1) + self.target_b1)
        # 输出层 output layer
        self.target_W2 = self.weight_variable([20, self.action_dim])
        self.target_b2 = self.bias_variable([self.action_dim])
        self.target_network_out = tf.nn.relu(tf.matmul(target_layer1_out, self.target_W2) + self.target_b2)

    def target_network_replace(self):
        tf.assign(self.target_W1, self.W1)
        tf.assign(self.target_b1, self.b1)
        tf.assign(self.target_W2, self.W2)
        tf.assign(self.target_b2, self.b2)
        # self.target_W1 = self.W1
        # self.target_b1 = self.b1
        # self.target_W2 = self.W2
        # self.target_b2 = self.b2
        print('update target network!')

    def network_loss_function(self):
        '''
        1.定义损失函数
        2.创建优化器
        :return:
        '''
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])
        # 通过神经网络算出了一个二维数组minibatch是32的话
        # network_out的shape是 32*3 分别代表选择三个动作的reward
        # 而action_input的shape是32*3，代表每32个sample选择的动作动
        # 作用的是一个数组表示，比如[0,0,1]代表像右走，两个数组相乘之后就得到获取的reward
        # 这个reward是通过神经网络计算而来的
        q_reward = tf.reduce_sum(tf.multiply(self.network_out, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_reward))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def percive(self, state, action, reward, next_state, done):
        '''
        将获取的环境的信息存储进队列中
        :param state: 状态
        :param action: 动作
        :param reward: 奖励
        :param next_state: 下一状态
        :param done: 游戏是否结束
        :return:
        '''
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.sample_deque.append((state, one_hot_action, reward, next_state, done))
        if len(self.sample_deque) > MAX_DEQUE_SIZE:
            self.sample_deque.popleft()
        if len(self.sample_deque) > BATCH_SIZE:
            self.train_network()




    def train_network(self):
        '''
        1.从包含一定数量样本的队列中抽取随机的mininbatch
        2.计算出样本对应的y值
        :return:
        '''
        minibatch = random.sample(self.sample_deque, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]   # state
        action_batch = [data[1] for data in minibatch]  # action
        reward_batch = [data[2] for data in minibatch]  # reward
        next_state_batch = [data[3] for data in minibatch]  # next_state
        # NatureDQN的主要改变的地方就是在计算y值(y_batch)的值的时候使用的是
        # target network来计算的。
        y_batch = []
        q_value_batch = np.zeros([32, 2])
        target_value = self.target_network_out.eval(feed_dict={self.target_state_input: next_state_batch})
        current_value = self.network_out.eval(feed_dict={self.state_input: next_state_batch})
        for j in range(0, BATCH_SIZE):
            q_value_batch[j][0] = np.max(target_value[j])
            q_value_batch[j][1] = np.max(current_value[j])
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))
        # 将获取的y值传入优化器，进行优化
        self.optimizer.run(feed_dict={self.y_input: y_batch,
                                      self.action_input: action_batch,
                                      self.state_input: state_batch})

    def weight_variable(self, shape):
        '''
        从截断的正态分布的中产生随机数，初始化变量weight_variable
        :param shape: 变量的shape
        :return: 初始化后的变量weight_variable
        '''
        init = tf.truncated_normal(shape)
        return tf.Variable(init)

    def bias_variable(self, shape):
        '''
        从截断的正态分布的中产生随机数，初始化变量bias_variable
        :param shape: 变量的shape
        :return: 初始化后的变量bias_variable
        '''
        init = tf.truncated_normal(shape)
        return tf.Variable(init)

    def action(self, state):
        '''
        Nature DQN与基本的DQN的区别就是使用
        当前的网络来选择动作,
        延迟的网络用来估计动作
        :param state: 当前状态
        :return: 根据现有的网络选择对应的动作
        '''
        return np.argmax(self.network_out.eval(feed_dict={self.state_input: [state]})[0])

    def choose_action(self, state):
        if self.epsilon > 0.001:
            self.epsilon -= (EPSILON-0.001)/100000
        if random.random() <= self.epsilon:
            rand_num = random.randint(0, self.action_dim - 1)
            return rand_num
        else:
            return self.action(state)

    def reload_network(self):
        '''
        加载已经训练了的神经网络参数
        :return:
        '''
        checkpoint = tf.train.get_checkpoint_state('./nets_files/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)


    def save_network(self,step):
        '''
        将已将训练过的神经网络参数进行保存，方便训练
        :return:
        '''
        save_path = self.saver.save(self.session, 'nets_files/mountain_car-{}'.format(step))
        print('saved net works!')


ENV_NAME = 'MountainCar-v0'
STEP = 200  # 每一次reset环境之前最多走的步数


def train(env):
    step_num = 0
    while 1:
        totle_reward = 0
        state = env.reset()
        for i in range(1, STEP+1):
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            position, velocity = next_state
            # 车开得越高 reward 越大,车最开始的位置是在 -0.5
            new_reward = abs(position - (-0.5))
            agent.percive(state, action, new_reward, next_state, done)
            state = next_state
            totle_reward += new_reward
            if done:
                break
        step_num += 1
        avg_reward = totle_reward/200
        print('train step:{}\t epsilon:{}\t avg_reward:{}'.format(step_num, agent.epsilon, avg_reward))
        if step_num % 100 == 0:
            agent.target_network_replace()
        if step_num % 1000 == 0:
            agent.save_network(step_num)
        if step_num >= 50000:
            exit(0)


def run(env):
    for j in range(100):
        totle_reward = 0
        state = env.reset()
        print('position:', state)
        for i in range(1, 201):
            env.render()
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            position, velocity = next_state
            # 车开得越高 reward 越大
            reward = abs(position - (-0.5))
            totle_reward += reward
            if done:
                print('step:', i)
                if i == 200:
                    print('failed!')
                    break
                else:
                    print("success!")
                    time.sleep(3)
                    break
        avg_reward = totle_reward / 200
        print('avg_reward:{}'.format(avg_reward))


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    agent = car_DQN(env)
    if sys.argv[1] == 'run':
        run(env)
    elif sys.argv[1] == 'train':
        train(env)
    else:
        print("commond error!")

