---
layout:       post
title:        "神经网络初试"
subtitle:     "分类与回归"
date:         2018-03-12 23:00:00
author:       "Jelliy"
header-img:   "img/port-bg-starry sky.jpg"
header-mask:  0.3
catalog:      true
multilingual: false
tags:
    - Neural Network
    - Deep Learning
    - Tensorflow
---

针对一个问题，解决的思路。

任务
1.对正弦曲线做回归

2.对两个正弦曲线做分类

方法
1. sklearn的机器学习方法做

2. 使用tensorflow，搭建神经网络，深度和宽度对结果的影响力。



先用第2种方法

论文参考
On the Number of Linear Regions of Deep Neural Networks


原理和实践
----

1.激活函数
2.损失函数

----

numpy实现

tensorflow实现（cpu 与 gpu）

目的
运行时间，内存的对比

1.搭建网络
2.调参
3.调整模型
4.优化

操作 x->y
f = wx+b
y = s(f)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

# 1.训练的数据
# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]


noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5# + noise
y_data=np.sin(x_data*10)

# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#logist 回归
#prediction = add_layer(xs, 1, 1, activation_function=tf.nn.relu)

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(xs, 1, 30, activation_function=tf.nn.relu)

# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, 30, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                    reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(10000):
   # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
   sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
   if i % 1000 == 0:
       # to see the step improvement
       print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, frameon=True)
plt.title("regression")

plt.scatter(x_data, y_data, label=u'actual value',edgecolor="blue",color="black", s=10)
# sess.run(a) or a.eval() a是tensor
#print(prediction.eval(session=sess))

prediction_value = sess.run(prediction,feed_dict = {xs:x_data})
plt.plot(x_data, prediction_value, label=u'predicted value',color="red")

#plt.gcf().autofmt_xdate()
plt.legend()

plt.show()





