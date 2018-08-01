---
layout:       post
title:        "线性回归"
subtitle:     "—— Iinear Regression"
date:         2018-07-27 15:18:00
author:       "Jelliy"
header-img:   "img/port-bg-starry sky.jpg"
header-mask:  0.3
catalog:      true
multilingual: false
tags:
    - 机器学习
---

## 数学基础
### 范数
范数是衡量某个向量空间（或矩阵）中的每个向量以长度或大小。范数的一般化定义：对实数p>=1， 范数定义如下:

$$||x||_p:=(\sum_i^n|x_i|^p)^{\frac{1}{p}}$$  

L1范数
当p=1时，是L1范数，其表示某个向量中所有元素绝对值的和。

L2范数
当p=2时，是L2范数， 表示某个向量中所有元素平方和再开根， 也就是欧几里得距离公式。

### 概率分布
#### 高斯分布(Gaussian distribution)

$$f(x|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

#### 拉普拉斯分布

$$f(x|\mu,b) = \frac{1}{2b}exp(-\frac{|x-\mu|}{b})$$ 

### 最大似然估计（MLE）

给定一堆数据，假如我们知道它是从某一种分布中随机取出来的，可是我们并不知道这个分布具体的参，即“模型已定，参数未知”。例如，我们知道这个分布是正态分布，但是不知道均值和方差；或者是二项分布，但是不知道均值。 最大似然估计（MLE，Maximum Likelihood Estimation）就可以用来估计模型的参数。MLE的目标是找出一组参数，使得模型产生出观测数据的概率最大:

$$\mathop{argmax}\limits_{\mu} p(X;\mu)$$

其中\\(p(X;\mu)\\)就是似然函数，表示在参数\\(\mu\\)下出现观测数据的概率。我们假设每个观测数据是独立的，那么有

$$p(x_1,x_2,\cdots,x_n;\mu)=\prod_{i=1}^n p(x_i;\mu)$$

为了求导方便，一般对目标取log。 所以最优化对似然函数等同于最优化对数似然函数：

$$\mathop{argmax}\limits_{\mu} p(X;\mu) = \mathop{argmax}\limits_{\mu} \log p(X;\mu)$$

### 最大后验概率估计（MAP）

假如这个参数\\(\mu\\)有一个先验概率,那么参数该怎么估计呢？
这就是MAP要考虑的问题。 MAP优化的是一个后验概率，即给定了观测值后使\\(\mu\\)概率最大:

$$\begin{align}
\mathop{argmax}\limits_{\mu} p(\mu|X) & = \mathop{argmax}\limits_{\mu} \frac{p(X|\mu)p(\mu)}{p(X)} \\
& = \mathop{argmax}\limits_{\mu} {p(X|\mu)p(\mu)} 
\end{align}$$

## 问题

m：训练集的样本个数 
n：训练集的特征个数（通常每行数据为一个x(0)=1与n个x(i) (i from 1 to n)构成，所以一般都会将x最左侧加一列“1”，变成n+1个特征） 
x：训练集（可含有任意多个特征，二维矩阵，行数m，列数n+1，即x0=1与原训练集结合） 
y：训练集对应的正确答案（m维向量，也就是长度为m的一维数组） 
h(x)：我们确定的模型对应的函数（返回m维向量） 
theta：h的初始参数（常为随机生成。n+1维向量）

## 线性回归

### 代价函数(Cost Function)

$$J(\theta_0,\theta_1\cdots\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})^2$$

### 推导

$$\theta = (X^TX)^{-1}X^Ty$$

## 线性回归概率解释
如果有数据集(X, Y)，并且Y是有白噪声（就是与测量得到的Y与真实的Y有均值为零的高斯分布误差），目的是用新产生的X来得到Y。如果用线性模型来测量，那么有:
其中 \\(X=(x_1, x_2...x_n)\\) ， \\(\epsilon\\) 是白噪声，即 \\(\epsilon \sim N(0, \delta^2)\\)。那么于一对数据集 \\((X_i, Y_i)\\) 来用，在这个模型中用\\(X_i\\) 得到 \\(Y_i\\) 的概率是 \\(Y_i \sim N(f(X_i), \delta^2)\\):

$$Y_i = f(X_i) + \epsilon_i$$

$$\begin{align}
P(Y_i|X_i,\theta) &= \frac{1}{\delta\sqrt{2\pi}}exp(-\frac{\epsilon_i}{2\delta^2}) \\
&= \frac{1}{\delta\sqrt{2\pi}}exp(-\frac{||f(X_i)-Y_i||^2}{2\delta^2})

\end{align}$$

假设数据集中每一对数据都是独立的，那么对于数据集来说由X得到Y的概率是：

$$P(Y|X,\theta)=\prod_{i}\frac{1}{\epsilon\sqrt{2\pi}}exp(-\frac{||f(X_i)-Y_i||^2}{2\delta^2})$$

根据决策论，就可以知道可以使概率 \\(P(Y\|X,\theta)\\) 最大的参数 \\(\theta^*\\) 就是最好的参数。那么我们可以直接得到最大似然估计的最直观理解：对于一个模型，调整参数 \\(\theta\\) ，使得用X得到Y的概率最大。那么参数 \\(\theta\\) 就可以由下式得到:

$$\begin{align}
\theta^* & = \mathop{argmax}\limits_{\theta}\prod_{i}\frac{1}{\epsilon\sqrt{2\pi}}exp(-\frac{||f(X_i)-Y_i||^2}{2\delta^2})\\
& = \mathop{argmax}\limits_{\theta}(-\frac{1}{2\delta}\sum_{i}||f(X_i)-Y_i||^2 + \sum_iln({\delta\sqrt{2\pi}}))\\
& = \mathop{argmax}\limits_{\theta}(\sum_{i}||f(X_i)-Y_i||^2)
\end{align}$$

### Ridge Regression 
Laplace先验导出L1正则化

先验的意思是对一种未知的东西的假设，比如说我们看到一个正方体的骰子，那么我们会假设他的各个面朝上的概率都是1/6，这个就是先验。但事实上骰子的材质可能是密度不均的，所以还要从数据集中学习到更接近现实情况的概率。同样，在机器学习中，我们会根据一些已知的知识对参数的分布进行一定的假设，这个就是先验。有先验的好处就是可以在较小的数据集中有良好的泛化性能，当然这是在先验分布是接近真实分布的情况下得到的了，从信息论的角度看，向系统加入了正确先验这个信息，肯定会提高系统的性能。我们假设参数$\theta$是如下的Laplace分布的，这就是Laplace先验：

$$P(\theta_i)=\frac{\lambda}{2}exp(-\lambda|\theta_i|)$$

其中\\(\lambda\\)是控制参数\\(\theta\\)集中情况的超参数，\\(\lambda\\)越大那么参数的分布就越集中在0附近。

在前面所说的最大似然估计事实上是假设了\\(\theta\\) 是均匀分布的，也就是 \\(P(\theta)=Constant\\) ，我们最大化的要后验估计，即是:

$$\begin{align}
\theta^* & = \mathop{argmax}\limits_{\theta}(\prod_{i}P(Y_i|X_i,\theta)\prod_iP(\theta_i))\\
 & = \mathop{argmax}\limits_{\theta}(\sum_{i}||f(X_i)-Y_i||^2 + \sum_iln(P(\theta_i)))\\
 & = \mathop{argmax}\limits_{\theta}(\sum_{i}||f(X_i)-Y_i||^2+\lambda\sum_i|\theta_i|)
\end{align}$$

### Lasso

### Elastic


## 梯度下降算法
常见的几种最优化方法（梯度下降法、牛顿法、拟牛顿法、共轭梯度法等）

## 正规方程、梯度下降的选择、比较

| 梯度下降 | 正规方程 |
| ------ | ------ | 
| 需要选择学习率 α | 一次运算得出 | 
| 当特征数量 n 大时也能较好适用 | 需要计算\\((X^TX)^{-1}\\)。如果特征数量 n 较大则运算代价大,因为矩阵逆的计算时间复杂度为 O(n 3 ),通常来说当 n 小于 10000 时还是可以接受的 | 

总结一下,只要特征变量的数目并不大,标准方程是一个很好的计算参数 θ 的替代方法。具体地说,只要特征变量数量小于一万,我通常使用标准方程法,而不使用梯度下降法。 
随着我们要讲的学习算法越来越复杂,例如,当我们讲到分类算法,像逻辑回归算法,我们会看到, 实际上对于那些算法,并不能使用标准方程法。对于那些更复杂的学习算法,我们将不得不仍然使用梯度下降法。因此,梯度下降法是一个非常有用的算法,可以用在有大量特征变量的线性回归问题。或者我们以后在课程中,会讲到的一些其他的算法,因为标准方程法不适合或者不能用在它们上。但对于这个特定的线性回归模型,标准方程法是一个比梯度下降法更快的替代算法。所以,根据具体的问题,以及你的特征变量的数量,这两种算法都是值得学习的。

## 代码实现 sklearn

## 造轮子，并与sklearn得出的结果比较，以及需要进一步优化地地方

## 参考

[线性回归(知乎)](https://www.zhihu.com/question/23536142)
