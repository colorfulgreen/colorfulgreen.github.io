---
layout: post
title:  "ML 笔记：神经网络"
date:   2020-07-15 09:30:00 +0800
categories: machine-learning 
---

目录
- [1 神经元模型](#神经元模型)
- [2 感知机与多层网络](#感知机与多层网络)
- [3 误差逆传播算法](#误差逆传播算法)
- [4 全局最小与局部极小](#全局最小与局部极小)
- [5 激活函数](#激活函数)
- [6 损失函数](#损失函数)
- [7 优化器](#优化器)

神经网络的定义[Kohonen, 1988]：

> 神经网络是由具有适应性的简单单元组成的广泛并行互联的网络，它的组织能够模拟生物神经系统对真实世界物体所作出的交互反应。

在机器学习中谈论神经网络时指的是“神经网络学习”，即机器学习与神经网络这两个学科领域的交叉部分。

## 神经元模型

神经网络中最基本的成分是 *神经元（neuron）*，每个神经元与其它神经元相连，当它兴奋时，会向相邻神经元发送化学物质，从而改变这些神经元内的电位；如果某神经元的电位超过了一个 *阈值（threshold）* 就会被激活，即兴奋，向其它神经元发送化学物质。

[McCulloch and Pitts, 1943] 将上述情形抽象为 *M-P 神经元模型*。该模型中，神经元接收来自其它 n 个神经元通过 **带权重的连接（connection）** 传递过来的输入信号，接收的总输入值将于神经元的阈值比较，然后通过 *激活函数（activation function）* 产生神经元的输出。

![mp-neuron](/assets/images/2007/mp-neuron.png#center){:width='480px'}

从计算机科学的角度看，只需将一个神经网络视为包含了许多参数的数学模型，这个模型由若干个函数，例如 $$ y_i = f(\sum_i w_i x_i - \theta_i) $$ 相互嵌套而得。有效的神经网络学习大多以数学证明为支撑。


## 感知机与多层网络

![perceptron](/assets/images/2007/perceptron.png#center){:width='360px'}

*感知机（Perceptron）* 由两层神经元组成，输入层接收外界输入信号后传递给输出层，输出层是 M-P 神经元，亦称 *阈值逻辑单元（threshold logic unit）*。

给定训练数据集，权重 $$w_i(i = 1,2,...,n)$$ 以及阈值 $$\theta$$ 可通过学习得到。阈值 $$\theta$$ 可看作是一个固定输入为 -1.0 的 *哑结点（dummy node）* 所对应的连接权重，这样，权重和阈值的学习可统一为权重的学习。对训练样例 $$(\mathbf x, y)$$，若当前感知机的输出为 $$\hat y$$，则感知机权重将这样调整：

$$ w_i \gets w_i + \Delta w_i $$,

$$ \Delta w_i = \eta (y - \hat y) x_i $$, 

其中 $$\eta \in (0,1)$$ 称为 *学习率（learning rate）*。显然，若感知机对训练样例预测正确，则感知机不发生变化，否则将根据错误的程度进行权重调整。

感知机只有输出层对神经元进行激活函数处理，即只有一层 *功能神经元（functional neuron）*，其学习能力非常有限。可以证明[Minsky and Papert, 1968]，若两类模式是线性可分的，则感知机的学习过程一定会收敛（converge）；否则感知机的学习过程将会发生震荡（fluctuation）。

![2-layer perceptron](/assets/images/2007/2-layer perceptron.png){:width="45%"}
![multi-layer networks](/assets/images/2007/multi-layer networks.png){:width="45%"}

要解决非线性可分问题，需使用多层功能神经元。如两层感知机可以解决异或问题，输出层与输入层之间的一层神经元被称为 *隐含层（hidden layer）*，隐含层和输出层都是拥有激活函数的功能神经元。


更一般的，常见的神经网络是所示的层级结构， **每层神经元与下一层神经元全互联** ，神经元之间不存在同层连接，也不存在跨层连接，这样的神经网络结构通常称为 *多层前馈神经网络（multi-layer feedforward neural networks）*。神经网络的学习过程，就是根据训练数据来调整神经元之间的 *连接权重（connection weight）*  以及每个功能神经元的阈值。

注意，“前馈” 并不意味着网络中信号不能向后传，而是指网络拓扑结构上不存在环或回路。

### 例：两层神经网络模型

* 用 Sigmoid, ReLU 作为激活函数
* 分类时用交叉熵作为损失函数

$$ \begin{aligned}
x = \text{raw input} \quad & & \\
z_1 = W_1^T x + b_{11} \quad & z_2 = W_2^T x + b_{12} & z_3 = W_3^T x + b_{13} \\
h_1 = ReLU(z_1) \quad & h_2 = ReLU(z_2) & h_3 = ReLU(z_3) \\
\theta_1 = U_1^T h_1 + b_{21} \quad & \theta_2 = U_2^T h_2 + b_{22} & \theta_3 = U_3^T h_3 + b_{23}
\end{aligned} 
\\
[\hat y_1, \hat y_2, \hat y_3] = softmax(\theta_1, \theta_2, \theta_3) \\ 
L_{ce}(\hat y, y) = -\sum_{j=1}^k y_j \log \hat y_j
$$



## 误差逆传播算法

逆误差传播（error BackPropagation，简称 BP）算法是迄今最成功的神经网络学习算法。需要指出，BP 算法不仅可用于多层前馈神经网络，还可用于其他类型的神经网络，如训练递归神经网络。但通常说 BP 网络时，一般指用 BP 算法训练的多层前馈神经网络。

给定训练集 $$ D = {(\bm x_1, \bm y_1), (\bm x_2, \bm y_2), \dots, (\bm x_m, \bm y_m)}, \bm x_i \in \mathbb R^d, \bm y \in \mathbb R^l $$，下图给出一个多层前馈神经网络结构。其中输出层第 j 个神经元的阈值用 $$\theta_j$$ 表示，隐层第 h 个神经元的阈值用 $$\gamma_h$$ 表示。

![BP](/assets/images/2007/BP.png#center){:width='480px'}

#### 网络参数更新

对样本 $$(\bm x_k, \bm y_k)$$，假定神经网络的输出为 $$\hat \bm y_k = (\hat y_1^k, \dots, \hat y_l^k)$$，即

$$ \hat y_j^k = f(\beta_j - \theta_j) $$

则网络在该样本上的均方误差为 (3.1)

$$ E_k = \frac 1 2 \sum_{j=1}^l (\hat y_j^k - y_j^k)^2 $$

**网络中有 $$(d + l)q + q + l$$ 个参数需要学习，包括不同层间的连接权、隐层及输出层的阈值。** BP 是一个迭代学习算法，在迭代的每一轮中采用广义的感知机学习规则对参数进行更新估计。参数的更新公式为：

$$  \Delta w_{hj}   = \eta g_j b_h \\
    \Delta \theta_j = -\eta g_j  \\
    \Delta v_{ih}   = \eta e_h x_i \\
    \Delta \gamma_h = -\eta e_h  $$

其中
                                   
$$ g_j = \hat y_j^k (1-\hat y_j^k)(y_j^k - \hat y_j^k) \\
   e_h = b_h(1-b_h) \sum_{j=1}^l w_{hj} g_j $$

学习率 $$ \eta \in (0,1) $$ 控制着算法每一步迭代中的更新步长。若太大则容易震荡，太小则收敛速度又会过慢。

这里以隐层到输出层的连接权 $$w_{hj}$$ 为例进行公式推导。对式 (3.1) 的均方误差 $$E_k$$，给定学习率 $$\eta$$，有

$$  \begin{aligned}
        \Delta w_{hj} & = -\eta \frac {\partial E_k} {\partial w_{hj}} \\
                      & = -\eta \frac {\partial E_k} {\partial \hat y_j^k} \frac {\partial \hat y_j^k} {\partial \beta_j} \frac{\partial \beta_j} {\partial w_{hj}}  \\
                      \\
        \frac{\partial \beta_j} {\partial w_{hj}} & = b_h \\
        \frac {\partial \hat y_j^k} {\partial \beta_j} & = f'(\beta_j - \theta_j)  \\
                & = f(\beta_j - \theta_j) [1 - f(\beta_j - \theta_j)] \text{ (利用 Sigmoid 函数的性质 f'(x)=f(x)(1-f(x)))} \\
                & = \hat y_j^k (1 - \hat y_j^k) \\
        \frac {\partial E_k} {\partial \hat y_j^k} & = \hat y_j^k - y_j^k 
    \end{aligned}
$$

#### BP 算法的工作流程

对每个训练样例，BP 算法执行以下操作：

* 先将输入示例提供给输入层神经元，然后逐层将信号前传，知道产生输出层的结果；
* 然后计算输出层的误差，再将误差逆向传播至隐层神经元，最后根据隐层神经元的误差来对连接权和阈值进行调整。

该迭代过程循环进行，直至达到某些停止条件为止，例如训练误差已经达到一个很小的值。

#### 累积误差逆传播算法

需注意的是，BP 算法的目标是最小化训练集 D 上的累积误差

$$ E = \frac 1 m \sum_{k=1}^m E_k $$

但上文介绍的参数更新规则是基于单个 $$E_k$$ 推导而得。如果类似地推导出基于累积误差最小化的更新规则，就得到了 *累积误差传播（accumulated error backpropagation）算法* 。

一般来说，标准 BP 算法每次更新只针对单个样例，参数更新地非常频繁，而且对不同样例进行更新的效果可能出现“抵消”现象。因此，为了达到同样的累积误差极小点，标准 BP 算法往往需进行更多次数的迭代。累积 BP 算法直接针对累积误差最小化，它在读取整个训练集 D 一遍后才对参数进行更新，其参数更新频率低得多。但在很多任务中，累积误差下降到一定程度之后，进一步下降会非常缓慢，这时标准 BP 往往会更快获得较好的解，尤其是在训练集 D 非常大时更明显。

#### BP 神经网络的层数

[Hornik et al., 1989] 证明，只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数。然而，如何设置隐层神经元的个数仍是个未决问题，实际应用中通常靠 *试错法（trial-by-error）* 调整。

#### 过拟合

由于其强大的表示能力，BP 神经网络经常遭遇过拟合。有两种常用策略来缓解 BP 网络的过拟合。

第一种策略是 *早停（early stopping）* 。将数据分成训练集和验证集，训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差，若训练集误差降低但验证集误差升高，则停止训练。

第二种策略是 *正则化（regularization）* [Barron, 1991; Girosi et al., 1995] ，其基本思想是在误差函数中增加一个用于描述网络复杂度的部分，例如连接权与阈值的平方和。仍令 $$E_k$$ 表示第 k 个训练样例上的误差，$$w_i$$ 表示连接权和阈值，则误差目标函数改变为

$$ E = \lambda \frac 1 m \sum_{k=1}^m E_k + (1-\lambda) \sum_i w_i^2 $$

其中 $$\lambda \in (0,1)$$ 用于对经验误差与网络复杂度这两项进行折中，常通过交叉验证法来估计。

## 全局最小与局部极小

神经网络在训练集上的误差 E 是关于连接权和阈值的函数。其训练过程可看作一个参数寻优过程，即在参数空间中，寻找一组最优参数使得误差最小。

基于梯度的搜索是使用最广泛的参数寻优方法。在此类方法中，我们从某些初始解出发，迭代寻找最优参数值。每次迭代中，我们先计算误差函数在当前点的梯度，然后根据梯度确定搜索方向。例如，由于负梯度方向是函数值下降最快的方向，因此梯度下降法就是沿着负梯度搜索最优解。若误差函数在当前点的梯度为零，则已达到局部极小，更新量将为零，这意味着参数的迭代更新将在此停止。如果误差函数有多个局部极小，则不能保证找到的解是全部最小，此时称参数寻优陷入了局部极小，这显然不是我们所希望的。

在现实任务中，人们常采用以下三种策略来试图“试图”跳出局部极小，从而进一步接近全部最小。

第一种策略，以多组不同参数值初始化多个神经网络，按标准方法训练后，取其中误差最小的解作为最终参数。这相当于从多个不同的初始点开始搜索，这样就可能陷入不同的局部极小，从中选择有可能获得最接近全局最小的结果。

第二种策略，使用 *模拟退火（simulated annealing）* 技术 [Aarts and Korst, 1989]. 模拟退火在每一步都以一定的概率接受比当前解更差的结果，从而有助于跳出局部极小。在每步迭代中，接受“次优解”的概率要随着时间的推移逐渐降低，从而保证算法稳定。

第三种策略，使用 *随机梯度下降* 。与标准梯度下降法精确计算梯度不同，随机梯度下降法在计算梯度时加入了随机因素。于是，即使陷入局部极小点，它计算出的梯度仍可能不为零，这样就有机会跳出局部极小继续搜索。

    Q：随机梯度下降在计算梯度时加入了什么随机因素？如果加入的随机因素足以使函数跳到其它局部极小，是否会引起震荡？

此外，*遗传算法（genetic algorithms）* [Goldberg, 1989] 也常用来训练神经网络以更好地逼近全局最小。需注意的是，上述用于跳出局部极小的技术大多是启发式，理论上尚缺乏保障。

## 激活函数

### 整流/修正线性单元ReLU

$$ ReLU(x) = max\{x, 0\} \\
   ReLU'(x) = 1[x>0] = sign(ReLU(x)) $$

## 损失函数

#### Softmax: 将给定的任意一组值映射成一个概率分布

$$ \sigma(\bf z) = [\frac {e^{z_1}} {\sum_{k=1}^K e_{z_k}}, 
                  \frac {e^{z_2}} {\sum_{k=1}^K e_{z_k}},
                  \dots,
                  \frac {e^{z_K}} {\sum_{k=1}^K e_{z_k}}] $$

![Softmax](/assets/images/2008/softmax.png#center){:width='200px'}

#### 负对数似然损失函数

$$ L(\bf y, f(\bf x, \theta)) = - \sum_{c=1}^C y_c \log f_c(\bf x, \theta) $$

例：对于一个三分类问题，真实类别为 [0, 0, 1]，预测的类别概率为 [0.3, 0.3, 0.4]，则

$$ L(\theta) = - (0 \times \log(0.3) + 0 \times \log(0.3) + 1 \times log(0.4)) = - \log(0.4) $$

## 优化器

### 梯度下降法

传统的优化方法是使用最小二乘法计算解析解，但有时面临着模型更复杂且样本量庞大的问题，当 **样本个数大于特征个数** 时，问题便转换为求解 **超定方程组** 的问题，相比使用最小二乘法求解大维数逆矩阵的方法，采用梯度迭代的梯度下降算法更具备优势。

梯度下降是指，在给定待优化的模型参数 $$ \theta \in R^d $$ 和目标函数 $$ J(\theta) $$ 后，算法通过沿梯度的相反方向更新参数 $$ \theta $$ 来最小化 $$ J(\theta) $$。对于每一个时刻t，我们可以用下述步骤描述梯度下降的流程。

1. 计算目标函数关于参数的梯度：

    $$ g_t = \Delta_\theta J(\theta) $$

2.  根据历史梯度计算一阶和二阶动量：

    $$ m_t = \phi(g_1, g_2, \dots, g_t) \\
       v_t = \psi(g_1, g_2, \dots, g_t) $$

3. 更新模型参数，其中 $$\epsilon$$ 为平滑项，防止分母为零，通常取 $$ 1e-8 $$.

$$ \theta_{t+1} = \theta_t - \frac 1 {\sqrt {v_t + \epsilon}} m_t $$

有三种梯度下降的变种：批梯度下降算法、随机梯度下降算法和小批量梯度下降算法。三种变种在计算优化目标时，所需的数据量不同。根据数据量的不同，我们需要在参数更新的精度和耗费时间两方面作出权衡。

#### 批梯度下降算法（Batch gradient descent, BGD）

批量梯度下降算法每次更新时，在整个训练集上计算损失函数关于参数 $$\theta$$ 的梯度：

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta) $$

因此，批梯度下降法的速度很慢。另外，批梯度下降无法处理超过内存容量限制的数据集。同样，批梯度下降算法也不能在线更新模型，即在运行过程中，不能增加新的样本。

在代码中，批量梯度下降算法看起来像这样：

    for i in range(nb_epochs):
        params_grad = evaluate_gradient(loss_function, data, params)
        params = params - learning_rate * params_grad

#### 随机梯度下降法（Stochastic gradient descent, SGD）

随机梯度下降算法根据每一条训练示例 $$x^{(i)}$$ 和标签 $$y^{(i)}$$ 更新参数：

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)}) $$

对于大数据集，因为批梯度下降法在每一个参数更新之前，会对相似的样本计算梯度，所以在计算过程中会有冗余。而SGD在每一次更新中只执行一次，从而消除了冗余。因而，通常SGD的运行速度更快，同时，可以用于在线学习。SGD以高方差频繁地更新，导致目标函数出现如下图所示的剧烈波动。


<figure>
  <img src="/assets/images/2008/sgd_fluctuation.png" width="240px"  />
  <figcaption>Image: SGD fluctuation (Source: <a href="https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png">Wikipedia</a>)</figcaption>
</figure>

与批梯度下降法的收敛会使得损失函数陷入局部最小相比，由于SGD的波动性，一方面，波动性使得SGD可以跳到新的和潜在更好的局部最优。另一方面，这使得最终收敛到特定最小值的过程变得复杂，因为SGD会一直持续波动。然而，已经证明当我们缓慢减小学习率，SGD与批梯度下降法具有相同的收敛行为，对于非凸优化和凸优化，可以分别收敛到局部最小值和全局最小值。与批梯度下降的代码相比，SGD的代码片段仅仅是在对训练样本的遍历和利用每一条样本计算梯度的过程中增加一层循环。注意，如6.1节中的解释，在每一次循环中，我们打乱训练样本。

    for i in range(nb_epochs):
        np.random.shuffle(data)
        for example in data:
            params_grad = evaluate_gradient(loss_function, example, params)
            params = params - learning_rate * params_grad

#### 小批量梯度下降法（Mini-batch gradient descent）

小批量梯度下降法最终结合了上述两种方法的优点，在每次更新时使用n个小批量训练样本：

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)}) $$

这种方法，a)减少参数更新的方差，这样可以得到更加稳定的收敛结果；b)可以利用最新的深度学习库中高度优化的矩阵优化方法，高效地求解每个小批量数据的梯度。通常，小批量数据的大小在50到256之间，也可以根据不同的应用有所变化。

在代码中，不是在所有样本上做迭代，我们现在只是在大小为50的小批量数据上做迭代：

    for i in range(nb_epochs):
        np.random.shuffle(data)
        for batch in get_batches(data, batch_size=50):
            params_grad = evaluate_gradient(loss_function, batch, params)
            params = params - learning_rate * params_grad

## 引用文献 

1. 周志华《机器学习》
2. [梯度下降优化算法综述](https://blog.csdn.net/google19890102/article/details/69942970)


