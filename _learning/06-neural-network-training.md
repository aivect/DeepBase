---
layout: article
title: "神经网络学习与梯度下降"
key: learning-06-neural-network-training
permalink: /learning/06-neural-network-training.html
sidebar:
  nav: learning
aside:
  toc: true
comment: true
mathjax: true
show_edit_on_github: false
show_subscribe: false
sharing: false
---

## 1.1 梯度下降中的两个关键问题

对于**凸函数而言，导数为0的点就是极小值点**，因此在数学中，我们常常**先对权重w求导，再令导数为0来求解极值**和对应的 。但是对于像神经网络这样的复杂模型，可能会有数百个w的存在，同时如果我们使用的是像交叉熵这样复杂的损失函数（机器学习中还有更多更加复杂的函数），令所有权重的导数为0并一个个 求解方程的难度很大、工作量也很大。因此我们转换思路，不追求一步到位，而使用迭代的方式逐渐接近损失函数的最小值。这就是优化算法的具体工作，优化算法的相关知识也都是关于“梯度下降目的：逐步迭代到损失函数最小值”的具体操作。

1、梯度下降的基本流程是什么，它是怎么找到损失函数的最小值的（写出流程）？ 2、梯度下降中是如何迭代权重 的（写出公式）？ 3、什么是梯度？什么是步长？

在模型训练中，梯度下降是指损失函数（或成本函数、目标函数）的值在下降。

1.1 梯度下降中的两个关键问题

观察SSE和交叉熵损失：
$$
\begin{aligned}
\text{CELoss}& s=-\sum_{i=1}^my_{i(k=j)}ln\sigma_i\quad\text{其中}\sigma=softmax(z),z=\boldsymbol{W}\boldsymbol{X} \\
\text{SSE}& =\frac1m\sum^m(z_i-\hat{z}_i)^2\quad\text{其中}z=\boldsymbol{X}\boldsymbol{w}
\end{aligned}
$$
任意损失函数中总是包含特征张量X，真实标签y或z，权重矩阵或权重向量w三个元素，其中特征张量X和真实标签来自于我们的数据，所以只要给定一组权重 w，就一定能够得出损失函数的具体数值。

![1691337100385]({{ '/assets/learning/06_neural_network_training/figure_008.png' | relative_url }})

在梯度下降最开始时，我们会先随机设定初始权重$w_{(0)} $，对应的纵坐标L(0)就是初始的损失函数值， 坐标点 $(w_{(0)}, L(0))$ 就是梯度下降的起始点。 接下来，我们从起始点开始，让自变量向损失函数减小最快的方向移动。

在这个过程中，每步的方向就是当前坐标点对应的梯度向量的反方向，每步的距离就是步长 * 当前坐标 点所对应的梯度向量的大小（也就是梯度向量的模长），梯度向量的性质保证了沿着其反方向、按照其 大小进行移动，就能够接近损失函数的最小值。

在这个过程中，存在两个关键问题：

- 1、怎么找出梯度向量的方向和大小？

- 2、怎么让坐标点按照梯度向量的反方向，移动与梯度向量大小相等的距离？

许多基于梯度下降算法的变化都是围绕这两个问题来进行。我们先来解决第一个问题。

### 1.1.1 找出梯度向量的方向和大小

梯度向量是多元函数上，各个自变量的偏导数组成的向量 。比如向量（1,2）的方向就是从原 点(0, 0)指向点(1, 2)的方向 。 梯度向量中的具体元素就是各个自变量的偏导数，这些偏导数的具体值必须依赖于当前所在 坐标点的值进行计算 。求梯度简写为$grad~L(w_1,w_2)\text{或者}\nabla L(w_1,w_2)。$

梯度向量中的具体元素就是各个自变量的偏导数，这些偏导数的具体值必须依赖于当前所在 坐标点的值进行计算 。

1）梯度的大小和方向对每个坐标点而言是独一无二的。坐标点一旦变化，梯度向量的方向和大小也会变 化。

2）每走到一个新的点，读取该点的坐标并带入梯度向量的表达式进行计算，就可以得到当前点对应的梯 度向量的方向和大小 。

### 1.1.2 一次迭代

假设我们的初始点是$(w_{1(0)},w_{2(0)}).$ ，梯度向量是$({\frac{\partial L}{\partial w_{1}}},{\frac{\partial L}{\partial w_{2}}})$ ，

为了控制每次走的距离的大小，我们将步长 （又叫做学习率）加入公式， 用 $t$代表走到t了第 步（即进行第 次迭代），则有：
$$
w_{(t+1)}=w_{(t)}-\eta\frac{\partial L}{\partial w}
$$
这就是我们迭代权重的迭代公式，其中偏导数的大小影响整体梯度向量的大小，偏导数前的减号影响整体梯度向量的方向 。一旦迭代 ，我们的损失函数也会发生变化。在整个损失函数的图像上，红色箭头就是梯度向 量的反方向，灰色箭头就是每个权重对应的迭代

![1691338129365]({{ '/assets/learning/06_neural_network_training/figure_009.png' | relative_url }})

**注**：梯度下降方向永远比loss函数低一个维度，也就是变量个数对应的维度，是自变量的变化方向，找到下一个自变量的位置，去确定这个位置上的loss值，再相对于原位置的loss值下降最快。

## 1.2 找出距离和方向：反向传播（为了简化求导过程）

### 1.2.1 反向传播的定义与价值

梯度向量是各个自变量求偏导后的表达式 再带入坐标点计算出来的，在这一步骤中，最大的难点在于如何获得梯度向量的表达式——也就是损失 函数对各个自变量求偏导后的表达式。在单层神经网络，例如逻辑回归（二分类单层神经网络）中，我 们有如下计算：

![1691369298046]({{ '/assets/learning/06_neural_network_training/figure_001.png' | relative_url }})

其中BCEloss是二分类交叉熵损失函数。在这个计算图中，从左向右计算以此的过程就是正向传播，因 此进行以此计算后，我们会获得所有节点上的张量的值（z、sigma以及loss）。根据梯度向量的定义， 在这个计算过程中我们要求的是损失函数对 的导数，所以求导过程需要涉及到的链路如下：

![1691369334151]({{ '/assets/learning/06_neural_network_training/figure_002.png' | relative_url }})
$$
\begin{aligned}
\frac{\partial Loss}{\partial w},\text{其中} \\
Loss& =-\sum_{i=1}^m(y_i*ln(\sigma_i)~+~(1-y_i)*ln(1-\sigma_i)) \\
&=-\sum_{i=1}^m(y_i*ln(\frac1{1+e^{-\boldsymbol{X}_i\boldsymbol{w}}})+(1-y_i)*ln(1-\frac1{1+e^{-\boldsymbol{X}_i\boldsymbol{w}}}))
\end{aligned}
$$
可以看出，已经很复杂了。 更夸张的是，在双层的、各层激活函数都是sigmoid的二分类神经网络上，更复杂

![1691369437210]({{ '/assets/learning/06_neural_network_training/figure_003.png' | relative_url }})

同样的，进行从左到右的正向传播之后，我们会获得所有节点上的张量。其中涉及到的求导链路如下 :

![1691369607540]({{ '/assets/learning/06_neural_network_training/figure_004.png' | relative_url }})
$$
\begin{aligned}
w^{(1\to2)}\text{我们有:}
&&\frac{\partial Loss}{\partial w^{(1\to2)}},\text{其中} \\
&&&\text{Loss} \begin{aligned}&=-\sum_{i=1}^m(y_i*ln(\sigma_i^{(2)})~+~(1-y_i)*ln(1-\sigma_i^{(2)}))\end{aligned} \\

&&&\begin{aligned}=&-\sum_{i=1}^m(y_i*ln(\frac{1}{1+e^{-\sigma_i^{(1)}w^{(1\to2)}}})~+~(1-y_i)*ln(1-\frac{1}{1+e^{-\sigma_i^{(1)}w^{(1\to2)}}}))\end{aligned} \\
&\text{对}w^{(0\to1)}\text{我们有} \\
&\frac{\partial Loss}{\partial w^{(0\to1)}},\text{其中}&& \text{Loss}\\
&\text{}&& =-\sum_{i=1}^m(y_i*ln(\sigma_i^{(2)})~+~(1-y_i)*ln(1-\sigma_i^{(2)})) \\
&&&=-\sum_{i=1}^m(y_i*ln(\frac1{1+e^{-\frac1{1+e^{-X_iw^{(0\to1)}}}w^{(1\to2)}}})+(1-y_i)*ln(1-\frac1{1+e^{-\frac1{1+e^{-X_iw^{(0\to1)}}}w^{(1\to2)}}}))
\end{aligned}
$$

求导过程的复杂是神经网络历史上的一大难题，这个难题直到 1986年才真正被解决。1986年，Rumelhart、Hinton和Williams提出了反向传播算法 （Backpropagation algorithm，又叫做Delta法则），利用链式法则成功实现了复杂网络求导过程的简 单化。（值得一提的是，多层神经网络解决XOR异或门问题是在1985年被提出的）

在高等数学中，存在着如下规则：
$$
\begin{aligned}\text{假设有函數}u&=h(z).\quad z=f(w)\text{,且两个函数在各自自变量的定义城上部可导,则有}z\\\frac{\partial u}{\partial w}&=\frac{\partial u}{\partial z}*\frac{\partial z}{\partial w}\end{aligned}
$$
**那就是中间激活值的梯度也需要计算**

当一个函数是由多个函数嵌套而成，最外层函数向最内层自变量求导的值， 等于外层函数对外层自变量求导的值 * 内层函数对内层自变量求导的值。这就是**链式法则**。

现在，因为Loss是一个内部嵌套了很多函数的函数，我们可以用链式法则将$\frac{\partial L_{oss}}{\partial w^{(1\to2)}}$ 拆解为如下结构：
$$
\frac{\partial Loss}{\partial w^{(1\to2)}}=\frac{\partial L(\sigma)}{\partial\sigma}*\frac{\partial\sigma(z)}{\partial z}*\frac{\partial z(w)}{\partial w}
$$
得到：
$$
\begin{aligned}
\frac{\partial Loss}{\partial w^{(1\to2)}}& =\frac{\partial L(\sigma)}{\partial\sigma}*\frac{\partial\sigma(z)}{\partial z}*\frac{\partial z(w)}{\partial w} \\
&=\frac{\sigma^{(2)}-y}{\sigma^2(1-\sigma^{(2)})}*\sigma^{(2)}(1-\sigma^{(2)})*\sigma^{(1)} \\
&=\sigma^{(1)}(\sigma^{(2)}-y)
\end{aligned}
$$
可以发现，将三个偏导数相乘之后，得到的最终的表达式其实非常简单。并且，其中所需要的数据都是 我们在正向传播过程中**已经计算出来的节点上的张量**。

同理，我们也可以得到对$w^{(0 \to1)}$ 的导数。本来我们需要求解 ：
$$
\begin{aligned}\frac{\partial Loss}{\partial w},\text{其中} \\Loss& =-\sum_{i=1}^m(y_i*ln(\sigma_i)~+~(1-y_i)*ln(1-\sigma_i)) \\&=-\sum_{i=1}^m(y_i*ln(\frac1{1+e^{-\boldsymbol{X}_i\boldsymbol{w}}})+(1-y_i)*ln(1-\frac1{1+e^{-\boldsymbol{X}_i\boldsymbol{w}}}))\end{aligned}
$$
现在根据链式法则，就有：
$$
\begin{gathered}
\text{por Tparmer ormar orela} \\
\frac{\partial Loss}{\partial w^{(0\to1)}}=\frac{\partial L(\sigma)}{\partial\sigma^{(2)}}*\frac{\partial\sigma(z)}{\partial z^{(2)}}*\frac{\partial z(\sigma)}{\partial\sigma^{(1)}}*\frac{\partial\sigma(z)}{\partial z^{(1)}}*\frac{\partial z(w)}{\partial w^{(0\to1)}} \\
\\
\text{其中前两项是在求解}w^{(1\to2)}\text{时求解过的,而后三项的求解结果都显而易见:} \\
=(\sigma^{(2)}-y)*\frac{\partial z(\sigma)}{\partial\sigma^{(1)}}*\quad\frac{\partial\sigma(z)}{\partial z^{(1)}}\quad*\frac{\partial z(w)}{\partial w^{(0\to1)}} \\
=(\sigma^{(2)}-y)*w^{1\to2}*(\sigma^{(1)}(1-\sigma^{(1)}))*X
\end{gathered}
$$
同样，这个表达式现在变得非常简单，并且，这**个表达式中所需要的全部张量，都是我们在正向传播中已经计算出来储存好的**，**或者再模型建立之初就设置好的**，因此在计算$w^{(0 \to1)}$的导数时，无需再重新计算如 $\sigma^{(2)}$这样的张量，这**就为神经网络计算导数节省了时间**。你是否注意到，我们是从右到左，从输出向输入，逐渐往前求解导数的表达式，并且我们所使用的节点上的张量，也是从后向前逐渐用到，这和 我们正向传播的过程完全相反。

这种**从右到左**，**不断使用正向传播中的元素对梯度向量进行计算的方式**，就是**反向传播**。

第一层神经元（或其他任意层）的梯度计算，只需要：

1. **后一层的梯度**（通过链式法则传递）。
2. **当前层的激活值或输入值**（来自前向传播保存的数据）。无需直接使用所有后续层的梯度或激活值。反向传播的递归性质使得计算高效且模块化。
3. 在反向传播中，**最后一层（输出层）的梯度计算仅需要前一层（倒数第二层）的激活值**，而不需要前面所有层的激活值。

### 1.2.2 PyTorch实现反向传

在梯度下降中，每走一步都需要更新梯度，所以计算量是巨大的。幸运的是，PyTorch 可以帮助我们自 动计算梯度，我们只需要提取梯度向量的值来进行迭代就可以了。在 PyTorch 中，我们有两种方式实现 梯度计算。一种是使用我们之前已经学过的AutoGrad。在使用AutoGrad时，我们可以使用 torch.autograd.grad()函数计算出损失函数上具体某个点/某个变量的导数，当我们需要了解具体某个点 的导数值时autograd会非常关键，比如：

```python
import torch
x = torch.tensor(1.,requires_grad = True) #requires_grad，表示允许对X进行梯度计算
y = x ** 2
torch.autograd.grad(y, x) #这里返回的是在函数y=x**2上，x=1时的导数值。
```

对于单层神经网络，autograd.grad会非常有效。但深层神经网络就不太适合使用grad函数了。对于深层神经网络，我们需要一次性计算大量权重对应的导数值，并且这些权重是以层为单位阻止成一个个权 重的矩阵，要一个个放入autograd来进行计算有点麻烦。所以我们会直接使 PyTorch 提供的基于 autograd的反向传播功能，lossfunction.backward()来进行计算。

注意：

- 在实现反向传播之前，**首先要完成模型的正向传播，并且要定义损失函数**。
- 反向传播一次后会释放动态计算图（动态计算图是正向传播动态生成的，释放为了节约内存。动态计算图（如 PyTorch 的 `grad_fn`）保存了前向传播的操作序列，使得反向传播时能按需计算局部梯度。），不能再继续执行第二次，若要再次反向传播，需要再次正向传播。

与可以重复进行的正向传播不同，一次正向传播后，反向传播只能进行一次。如果希望能够重复进行反向传播，可以在进行第一次反向传播的时候加上参数.backward(retain_graph=True)。对于保存中间节点的梯度用 张量.retain_grad()

```python
#导入库、数据、定义神经网络类，完成正向传播
#继承nn.Module类完成正向传播
import torch
import torch.nn as nn
from torch.nn import functional as F
#确定数据
torch.manual_seed(420)
X = torch.rand((500,20),dtype=torch.float32) * 100

# 使用交叉熵函数CEL需要y为一维的
y = torch.randint(low=0,high=3,size=(500,1),dtype=torch.float32)

#定义神经网路的架构
"""
注意：这是一个三分类的神经网络，因此我们需要调用的损失函数多分类交叉熵函数CEL
CEL类已经内置了sigmoid功能，因此我们需要修改一下网络架构，删除forward函数中输出层上的sigmoid
函数，并将最终的输出修改为zhat
"""
class Model(nn.Module):
    def __init__(self,in_features=10,out_features=2):
        super(Model,self).__init__() #super(请查找这个类的父类，请使用找到的父类替换现在
        的类)
        self.linear1 = nn.Linear(in_features,13,bias=True) #输入层不用写，这里是隐藏
        层的第一层
        self.linear2 = nn.Linear(13,8,bias=True)
        self.output = nn.Linear(8,out_features,bias=True)
    def forward(self, x):
        ''' sigma1 = torch.relu(self.linear1(x))
        sigma2 = torch.sigmoid(self.linear2(sigma1))
        zhat = self.output(sigma2)'''
        z1 = self.linear1(x)
        sigma1 = torch.relu(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        z3 = self.output(sigma2)
        #sigma3 = F.softmax(z3,dim=1)
        return z3
input_ = X.shape[1] #特征的数目
output_ = len(y.unique()) #分类的数目
#实例化神经网络类
torch.manual_seed(420)
net = Model(in_features=input_, out_features=output_)
#前向传播
zhat = net.forward(X)
#定义损失函数
criterion = nn.CrossEntropyLoss()
#对打包好的CorssEnrtopyLoss而言，只需要输入zhat
loss = criterion(zhat,y.reshape(500).long())
loss
net.linear1.weight.grad #不会返回任何值
#反向传播，backward是任意损失函数类都可以调用的方法，对任意损失函数，backward都会求解其中全部w的梯度
loss.backward()
net.linear1.weight.grad #返回相应的梯度
#与可以重复进行的正向传播不同，一次正向传播后，反向传播只能进行一次
#如果希望能够重复进行反向传播，可以在进行第一次反向传播的时候加上参数retain_graph
loss.backward(retain_graph=True)
loss.backward()

```

唯一需要说明的点是，在使用autograd的时 候，我们强调了requires_grad的用法，但在定义打包好的类以及使用loss.backward()的时候，我们却 没有给任何数据定义requires_grad=True。这是因为：

1、当使用nn.Module继承后的类进行正向传播时，我们的权重 是自动生成的，在生成时就被自动设置 为允许计算梯度（requires_grad=True），所以不需要我们自己去设置

2、同时，观察我们的反向传播过程：

![1691369607540]({{ '/assets/learning/06_neural_network_training/figure_004.png' | relative_url }})

不难发现，我们的**特征张量 与真实标签 都不在反向传播的过程**当中，但是 与 其实都是损失函数计算需要用的值，在计算图上，这些值都位于叶子节点上，我们在定义损失函数时，并没有告诉**损失函数 哪些值是自变量，哪些是常数**，那backward函数是怎么判断具体求解哪个对象的梯度的呢？

其实就是靠requires_grad。首先**backward值会识别叶子节点，不在叶子上的变量是不会被backward考 虑的**。对于全部叶子节点来说，只有属性requires_grad=True的节点，才会被计算。在设置X与y 时， 我们都没有写requires_grad参数，也就是默认让“允许求解梯度”这个选项为False，所以backward在计 算的时候就只会计算关于w的部分。

我们也可以将 和 或者任何除了权重以及截距的量的requires_grad打开，一旦我们设置为 True，backward就会在帮助我们计算 的导数的同时，也棒我们计算以 或 为自变量的导数。在正常 的梯度下降和反向传播过程中，我们是不需要这些导数的，因此我们一律不去管requires_grad的设置， 就让它默认为False，以节约计算资源。当然，如果你的 是自己设置的，千万记得一定要设置 requires_grad=True。

## 1.3 移动坐标点

### 1.3.1 走出第一步

有了大小和方向，接下来我们就可以开始走出我们的第一步了。来看权重的迭代公式：
$$
w_{(t+1)}=w_{(t)}-\eta\frac{\partial L}{\partial w}
$$
现在我们的偏导数部分已经计算出来了，就是我们使用backward求解出的结果。而 学习率，或者步 长，是我们在迭代开始之前就人为设置好的，一般是0.01~0.5之间的某个小数。因此现在我们已经可以 无障碍地使用代码实现权重的迭代了。

步长/学习率：英文是learning rate，所以常常简写为lr

```python
lr = 10
dw = net.linear1.weight.grad
w = net.linear1.weight.data
#对任意w可以有
w -= lr * dw
```

**获取weight张量**：

```python
net.linear1.weight.data
```

**获取weight张量梯度**

```python
net.linear1.weight.grad
```

**普通梯度下降**就是在重复 {**正向传播、计算梯度、更新权重**(损失函数坐标点改变)} 的过程，但这个过程往往非常漫长。 无论如何，梯度下降都是一个缓慢的过程。在这个基础上，我们提出了加速迭代的数个方 法，其中一个很关键的方法，就是使用动量Momentum。

### 1.3.2 从第一步到第二步：动量法Momentum

之前我们说过，在梯度下降过程中，起始点是一个“盲人”，它看不到也听不到全局，所以我们每移动一次都要重新计算方向与距离，并且每次只能走一小步。但不只限于此，起始点不仅看不到前面的路，也 无法从过去走的路中学习。

真正高效的方法是：在历史方向与现有方向相同的情况下，迈出大步子，在历史方向与现有方向相反的情况下，迈出小步子。那要怎么才能让起始点了解过去的方向呢？**我们让上一步的梯度向量与现在这一点的梯度向量以加权的方式求和，求解出受到上一步大小和方向影响的真实下降方向，再让坐标点向真实下降方向移动**。在坐标轴上，可以表示为：

![1691387734644]({{ '/assets/learning/06_neural_network_training/figure_005.png' | relative_url }})

其中，对上一步的梯度向量加上的权重被称为**动量参数**（也叫做衰减力度，通常使用**$\gamma$**进行表示），对这 一点的梯度向量加上的权重就是步长（依然表示为 $\eta$），真实移动的向量为v ，被称为”动量 “（Momentum）。将上述过程使用公式表示，则有：

下面公式的$-\eta$这部分的负号就是表示梯度的反方向！
$$
\begin{aligned}v_{(t)}&=\gamma v_{(t-1)}-\eta\frac{L}{\partial\boldsymbol{w}}\\
w_{(t+1)}&=\boldsymbol{w}_{(t)}+v_{(t)}\text{：注意是加号}\end{aligned}
$$
在第一步中，没有历史梯度方向，因此第一步的真实方向就是起始点梯度的反方向，$v_0 = 0$ 。其中 **$v_{(t-1)}$代表了之前所有步骤所累积的动量和**。在这种情况下，梯度下降的方向有了“惯性”，受到历史累计 动量的影响，当新坐标点的梯度反方向与历史累计动量的方向一致时，历史累计动量会加大实际方向的 步子，相反，当新坐标点的梯度反方向与历史累计动量的方向不一致时，历史累计动量会减小实际方向 的步子。

```python
#恢复小步长
lr = 0.1
# 自定义动量参数
gamma = 0.9
dw = net.linear1.weight.grad
w = net.linear1.weight.data
v = torch.zeros(dw.shape[0],dw.shape[1])
#v要能够跟dw相减，因此必须和dw保持相同的结构，初始v为0，但后续v会越来越大

```

#分割cell，不然重复运行的时候w会每次都被覆盖掉：

```python
#对任意w可以有
v = gamma * v - lr * dw
w -= v
w
#不难发现，当加入gamma之后，即便是较小的步长，也可以让w发生变化
```

### 1.3.3 torch.optim实现带动量的梯度下降

在 PyTorch 库的架构中，拥有专门实现优化算法的模块torch.optim。我们在之前的课程中所说的迭代流程，都可以通过torch.optim模块来简单地实现 ，每一步step()都会对所有层上的w做动量梯度下降。

基于之前定义的类Model来实现梯度下降的一轮迭代：

```python
# 1、导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# 2、确定数据、确定训练前就需要设置的值（超参数设置）
lr = 0.1
gamma = 0.9
torch.manual_seed(420)
X = torch.rand((500,20),dtype=torch.float32) * 100
y = torch.randint(low=0,high=3,size=(500,1),dtype=torch.float32)
input_ = X.shape[1] #特征的数目
output_ = len(y.unique()) #分类的数目

# 3、定义神经网路的架构
class Model(nn.Module):
    def __init__(self,in_features=10,out_features=2):
        super(Model,self).__init__() #super(请查找这个类的父类，请使用找到的父类替换现在
        的类)
        self.linear1 = nn.Linear(in_features,13,bias=True) #输入层不用写，这里是隐藏
        层的第一层
        self.linear2 = nn.Linear(13,8,bias=True)
        self.output = nn.Linear(8,out_features,bias=True)

    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.relu(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        z3 = self.output(sigma2)
        #sigma3 = F.softmax(z3,dim=1)
        return z3

# 4、实例化神经网络，调用优化算法需要的参数
torch.manual_seed(420)
net = Model(in_features=input_, out_features=output_)
net.parameters()

# 5、定义损失函数，实例化
criterion = nn.CrossEntropyLoss()

# 6、定义优化算法。optim.SGD是随机梯度下降
opt = optim.SGD(
 net.parameters() #要优化的参数是哪些？
 , lr=lr #学习率
 , momentum = gamma #动量参数
 )
```

接下来开始进行一轮梯度下降：

opt.step()：更新权重w，从这一瞬间开始，坐标点就发生了变化，所有的梯度必须重新计算。因为经过loss.backward()反向传播，则得到了每个样本特征的梯度。

```python
# 1、向前传播，得到输出层的输出结果加和zhat
zhat = net.forward(X)
# 2、计算损失函数
loss = criterion(zhat,y.reshape(500).long())
# 3、反向传播
loss.backward()
# 4、走一步。更新权重w，从这一瞬间开始，坐标点就发生了变化，所有的梯度必须重新计算。因为经过loss.backward()反向传播，则得到了每个样本特征的梯度。
opt.step()
'''后续需要继续正向传播'''
# 5、清除原来储存好的，基于上一个坐标点计算的梯度，为下一次计算梯度腾出空间
opt.zero_grad()

print(loss)
print(net.linear1.weight.data[0][:10])
```

## 1.4 开始迭代：batch_size与epoches

### 1.4.1 为什么要有小批量？

在实现一轮梯度下降之后，只要在梯度下降的代码外面加上一层循环，就可以顺利实现迭代多次的梯度下降了。为了提升梯度下降的速度，我们在使用了动量法，同时，我 们也要在使用的数据上下功夫。

在深度学习的世界中，神经网络的训练对象往往是图像、文字、语音、视频等非结构化数据，这些数据 的特点之一就是特征张量一般都是大量高维的数据。比如在深度学习教材中总是被用来做例子的入门级 数据MNIST，其训练集的结构为(60000,784)。对比机器学习中的入门级数据鸢尾花（结构为(150,4)）, 两者完全不在一个量级上。在深度学习中，如果梯度下降的每次迭代都使用全部数据，将会非常耗费计 算资源，且样本量越大，计算开销越高。虽然 PyTorch 被设计成天生能够处理巨量数据，但我们还是需 要在数据量这一点上下功夫。这一节，我们开始介绍**小批量随机梯度下降**（mini-batch stochastic gradient descent，简写为mini-batch SGD）。

小批量随机梯度下降是深度学习入门级的优化算法（梯度下降是入门级之下的），其求解与迭代流程与 传统梯度下降（GD）基本一致，不过二者在迭代权重时使用的数据这一点上存在巨大的不同。传统梯度 下降在每次进行权重迭代（即循环）时都使用全部数据，每次迭代所使用的数据也都一致。而minibatch SGD是每次迭代前都会从整体采样一批固定数目的样本组成批次（batch） ，并用 中的样本进 行梯度计算，以减少样本量。

为什么会选择mini-batch SGD作为神经网络的入门级优化算法呢？有两个比较主流的原因。第一个是， 比起传统梯度下降，mini-batch SGD更可能找到全局最小值。

当走到一点时，可能再向所有方向走，都会增加，不再下降，则认为是全局最优，但可能只是局部最优。

<img src="{{ '/assets/learning/06_neural_network_training/figure_010.png' | relative_url }}" alt="1691390375311" style="zoom:50%;" />

尽可能找到全局最优一直都是优化算法的目标。为什么说mini-batch SGD更容易找到全局最优呢？

<img src="{{ '/assets/learning/06_neural_network_training/figure_006.png' | relative_url }}" alt="1691390581495" style="zoom:50%;" />

传统梯度下降是每次迭代时都**使用全部数据的梯度下降**，所以每次使用的数据是一致的，因此梯度向量 的方向和大小都只受到权重 的影响，所以梯度方向的变化相对较小，很多时候看起来梯度甚至是指向 一个方向（如上图所示）。这样带来的优势是可以使用较大的步长，快速迭代直到找到最小值。但是缺 点也很明显，由于梯度方向不容易发生巨大变化，所以一旦在迭代过程中落入局部最优的范围，传统梯 度下降就很难跳出局部最优，再去寻找全局最优解了。

而mini-batch SGD在每次迭代前都会随机抽取一批数据，所以每次迭代时带入梯度向量表达式的数据是不同的，**梯度的方向同时受到系数w和b带入的训练数据的影响**，因此每次迭代时梯度向量的方向都会 发生较大变化。并且，当抽样的数据量越小，本次迭代中使用的样本数据与上一次迭代中使用的样本数 据之间的差异就可能越大，这也导致本次迭代中梯度的方向与上一次迭代中梯度的方向有巨大差异。所 以对于mini-batch SGD而言，它的梯度下降路线看起来往往是曲折的折线（如上图所示）。

极端情况下，当我们每次随机选取的批量中只有一个样本时，梯度下降的迭代轨迹就会变得异常不稳定 （如上图所示）。我们称这样的梯度下降为随机梯度下降（stochastic gradient descent，SGD）。

mini-batch SGD的优势是算法不会轻易陷入局部最优，由于每次梯度向量的方向都会发生巨大变化，因此一旦有机会，算法就能够跳出局部最优，走向全局最优（当然也有可能是跳出一个局部最优，走向另 一个局部最优）。不过**缺点**是，需要的迭代次数变得不明。如果最开始就在全局最优的范围内，那可能 只需要非常少的迭代次数就收敛，但是如果最开始落入了局部最优的范围，或全局最优与局部最优的差 异很小，那可能需要花很长的时间、经过很多次迭代才能够收敛，毕竟不断改变的方向会让迭代的路线 变得曲折。

为了mini-batch SGD这“不会轻易被局部最优困住”的优点，我们在神经网络中使用它作为 优化算法（或优化算法的基础）。当然，还有另一个流传更广、更为认知的理由支持我们使用mini batch SGD：mini-batch SGD可以提升神经网络的计算效率，让神经网络计算更快。

### 1.4.2 batch_size与epoches

- batch_size

在mini-batch SGD中，我们选择的批量batch含有的样本数被称为batch_size，批量尺寸，这个尺寸一 定是小于数据量的某个正整数值。每次迭代之前，我们需要从数据集中抽取batch_size个数据用于训练。

在普通梯度下降中，因为没有抽样，所以每次迭代就会将所有数据都使用一次，迭代了t次时，算法就将数据学习了t次。可以想象，对同样的数据，算法学习得越多，也有应当对数据的状况理解得越深，也就 学得越好。然而，并不是对一个数据学习越多越好，毕竟学习得越多，训练时间就越长，同时，我们能 够收集到的数据只是“样本”，并不能够代表真实世界的客观情况。例如，我们从几万张猫与狗的照片中 学到的内容，并不一定就能适用于全世界所有的猫和狗。如果我们的照片中猫咪都是有毛的，那神经网 络对数据学习的程度越深，它就越有可能认不出无毛猫。

因此，虽然我们希望算法对数据了解很深，但 我们也希望算法不要变成”书呆子“，要保留一些灵活性（**保留一些泛化能力**）。关于这一点，我们之后会详细展开来说明，但大家现在需要知道的是，**算法对同样的数据进行学习的次数并不是越多越好**。

在mini-batch SGD中，因为每次迭代时都只使用了一小部分数据，所以它迭代的次数并不能代表全体数 据一共被学习了多少次。所以我们需要另一个重要概念：epoch，读音/ˈepək/，来定义全体数据一共被 学习了多少次。

- epoch

epoch是衡量训练数据被使用次数的单位，一个epoch表示优化算法将全部训练数据都使用了一 次。它与梯度下降中的迭代次数有非常深的关系，我们常使用“完成1个epoch需要n次迭代“这样的 语言。

假设一个数据集总共有m 个样本，我们选择的batch_size是 $N_B$，即每次迭代时都使用 $N_B$个样本，则一 个epoch所需的迭代次数的计算公式如下：
$$
\text{完成一个}epoch\text{所需要的迭代次数}n=\frac m{N_B}
$$
在深度学习中，我们常常定义num_epoches作为梯度下降的最外层循环，batch_size作为内层循环。有 时候，我们希望数据被多学习几次，来增加模型对数据的理解。有时候，我们会控制模型对数据的训 练。总之，我们会使用epoch和batch_size来控制训练的节奏。

### 1.4.3 TensorDataset与DataLoader

![]({{ '/assets/learning/06_neural_network_training/figure_007.png' | relative_url }})

要使用小批量随机梯度下降，我们就需要对数据进行采样、分割等操作。在 PyTorch 中，操作数据所需 要使用的模块是torch.utils，其中utils.data类下面有大量用来执行数据预处理的工具。

在MBSGD中，我们需要将数据划分为许多组**特征张量+对应标签**的形式，因此最开始我们要将**数据的特征张量与标签打包成一个对象**。

- **TensorDataset**(*张量)

在MBSGD中，我们需要将数据划分为许多组特征张量+对应标签的形式，因此最开始我们要将数据的特征张量与标签打包 成一个对象。 必须张量的第一维度需要相等个，才可以合并。合并两个张量和并结果为 **元组**：**(tensor([[……]]), tensor({……))**这样。**这些张量的N-1维张量合并为一个元组的中一个元素**，所以张量的第一维度必须相等，才可以合并成功。

```python
import torch
from torch.utils.data import TensorDataset
a = torch.randn(500,2,3)
b = torch.randn(500,3,4,5)
c = torch.randn(500,1)
TensorDataset(a,b,c)[0]
#试试看合并a与c，我们一般合并特征张量与标签，就是这样合并的
for x in TensorDataset(b,c):#generator
 print(x)
 break
'''(tensor([[[ 0.3641, 1.1461, 1.3315, -0.6803, -0.1573],
 [-0.3813, 0.0569, 1.4741, -0.2237, 0.4374],
 [ 0.4338, 0.7315, -0.2749, 0.0160, -0.2451],
 [-0.5867, -0.5889, 1.8905, -0.7718, -1.7899]],

 [[-0.4048, 0.7898, 0.3773, 0.7166, 0.0490],
 [-0.9121, -0.0489, -0.8179, -1.8548, -0.3418],
 [ 0.0873, 0.3071, -0.9272, 1.4546, -0.8360],
 [ 1.2235, 1.2197, -0.5222, 0.2297, -0.8180]],

 [[ 0.4578, -2.0396, -0.1589, -0.3033, -0.6102],
 [ 1.1299, 0.8919, -0.5627, 0.4364, -0.2321],
 [ 0.1634, 1.4667, -0.7651, -0.6503, 0.0228],
 [ 0.8123, 0.9057, 1.3573, -0.3826, 0.2580]]]), tensor([-0.4818]))'''
```

- **两个DataLoader**(data, batch_size, shuffle, drop_last=False)

**shuffle=True #划分小批量之前请随机打乱我们的数据。因此drop_last = True舍弃最后一个batch样本不够组成一个batch_size时的batch时，在下一次的epoch迭代仍可能训练到。**

当我们经过 TensorDataset将数据打包成一个对象之后，我们需要使用划分小批量的功能DataLoader。DataLoader是处理训练前专用的功能，它可以接受任意形式的数组、张量作为输入，并把他们一次性转换为神经网络可以 接入的tensor。

对于DataSet合并对象数据和标签，经过DataLoader，会由:

元组：(tensor([[……]]), tensor({……))

----> 列表：[tensor([[……]]), tensor([……])]

```python
data = TensorDataset(b,c)
dataset = DataLoader(data
 , batch_size = bs
 , shuffle = True #划分小批量之前请随机打乱我们的数据
 , drop_last = True #你要舍弃最后一个batch吗？
 )
```

**相关属性**：

```python
len(dataset) #一共有多少个batch
dataset.dataset#展示里面全部的数据
len(dataset.dataset) #展示里面全部的数据个数

dataset.dataset[0] #全部数据中的单个样本
dataset.dataset[0][0] #单个样本的数据张量
dataset.dataset[0][1] #单个样本的 标签

dataset.batch_size #查看现有的batch_size
```

将外部导入的数据放到 TensorDataset与DataLoader里来使用：

```python
#从pandas导入数据
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Pythonwork\DEEP LEARNING\WEEK
3\Datasets\creditcard.csv")
data.shape
data.head()
X = torch.tensor(np.array(data.iloc[:,:-1]),dtype=torch.float32)
y = torch.tensor(np.array(data.iloc[:,-1]),dtype=torch.float32)
data = TensorDataset(X,y)
batchdata = DataLoader(data, batch_size=1000,shuffle=True)
for x,y in batchdata:
 print(x)

#从pandas导入数据
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Pythonwork\DEEP LEARNING\WEEK
3\Datasets\creditcard.csv")
data.shape
data.head()
X = torch.tensor(np.array(data.iloc[:,:-1]),dtype=torch.float32)
y = torch.tensor(np.array(data.iloc[:,-1]),dtype=torch.float32)
data = TensorDataset(X,y)
batchdata = DataLoader(data, batch_size=1000,shuffle=True)
for x,y in batchdata:
 print(x)

```

## 1.5 在MINST-FASHION上实现神经网络的学习流程

了神经网络使用小批量随机梯度下降进行迭代的流程，现在我们要整合本节课中所有的 代码实现一个完整的训练流程。首先要梳理一下整个流程：

0）导库

1）设置超参数：步长 ，动量值 ，迭代次数 ，batch_size等信息，（如果需要）设置初始权重 ，

2）导入数据，将数据切分成batches

3）定义神经网络架构

4）定义损失函数 ，如果需要的话，将损失函数调整成凸函数，以便求解最小值

5）定义所使用的优化算法

6）开始在epoches和batch上循环，执行优化算法：

- 6.1）调整数据结构（TensorDataSet、DataLoader），确定数据能够在神经网络、损失函数和优化算法中顺利运行

- 6.2）完成向前传播，计算初始损失 loss = criterion(zhat,y.reshape(500).long())

- 6.3）利用反向传播，在损失函数$L(W)$上对每一个w求偏导数 loss.backward()

- 6.4）迭代当前权重opt.step()

- 6.5）清空本轮梯度 opt.zero_grad()

- 6.6）完成模型进度与效果监控

7）输出结果

dataloader、TensorDataset - 对数据的结构、归纳方式进行变换

torchvision.transforms - 对数据集的数字本身进行修改

这次我们要使用 PyTorch 中自带的数据，MINST-FATIO ：

图像数据集的特征张量结构：**(sample_size, H- height, W - width, C - color)，sample_size图像索引，H- height, W - width为横纵像素大小，C - color颜色通道为4(rgb+alpha(透明度))/3(rgb)/1(灰)。**

mnist.targets.unique()：查看类别

mnist.classes：查看类的意思

mnist.data : 查看特征张量

mnist.targets :查看标签

matplotlib图像显示需要将tensor转为numpy列表才可以。

```python
'''1 导库，设置各种初始值'''
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

#确定数据、确定优先需要设置的值
lr = 0.15
gamma = 0
epochs = 10
bs = 128

```

- 2 导入数据，分割小批量

```python
'''2 导入数据，分割小批量'''
#实例化数据
mnist = torchvision.datasets.FashionMNIST(
 root = "D:\01Postgraduate\study_code\FashionMNIST" #你的计算机上的某个目录
, download = False #是否下载数据集
, train = True # 是否用来训练，训练需要大量数据集
, transform = transforms.ToTensor() # 将数据集转化为 Tensor 结构
)
#放入进行迭代的数据结构是什么样的？
batchdata = DataLoader(mnist,batch_size=bs, shuffle = True)
```

- 对于二维(500,20) - 神经网络的第一层的输入神经元个数20

- 对于x - 四维（128,1,28,28）转为（128,1x28x28），在正向传播forward(self，x)作为x，在传入之前不需要降维，但是进入forward必须降维。神经网络的第一层的输入神经元个数为1x28x28（可有numel()获取到所有零维张量个数）。因为nn.linear(输入神经元个数、输出神经元个数)，对于激活函数ReLu需要输入为二维的数据，

```python
# 神经网络的输入数据结构
input_ = mnist.data[0].numel() #特征的数目，一般是第一维之外的所有维度相乘的数
output_ = len(mnist.targets.unique()) #分类的数目
```

- 3 定义神经网络的架构

x = x.view(-1, 28*28)：-1做占位符，PyTorch 自动计算

```python
'''3 定义神经网络的架构'''
class Model(nn.Module):
    def __init__(self,in_features=10,out_features=2):
        super().__init__()
        #self.normalize = nn.BatchNorm2d(num_features=1)
        self.linear1 = nn.Linear(in_features,128,bias=False)
        self.output = nn.Linear(128,out_features,bias=False)

    def forward(self, x):
        #x = self.normalize(x)
        x = x.view(-1, 28*28)
        #需要对数据的结构进行一个改变，这里的“-1”代表，我不想算，请 PyTorch 帮我计算
        sigma1 = torch.relu(self.linear1(x))
        z2 = self.output(sigma1)
        sigma2 = F.log_softmax(z2,dim=1)
        return sigma2
```

- 4 定义训练函数

**训练函数：包含了损失函数、优化算法、以及梯度下降整个流程**。

```python
def fit(net,bacthdata,lr=0.01, epochs=5, gamma = 0):
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(),lr=lr,momentum = gamma)
    correct = 0 #循环开始之前，预测正确的值为0
    samples = 0 #循环开始之前，模型一个样本都没有见过
    for epoch in range(epochs): #全数据被训练几次
 		for batch_idx,(x,y) in enumerate(batchdata):
             y = y.view(x.shape[0]) #降维，一维
             sigma = net.forward(x) #正向传播
             loss = criterion(sigma,y)
             loss.backward()
             opt.step()
             opt.zero_grad()

 #求解准确率，全部判断正确的样本数量/已经看过的总样本量
 yhat = torch.max(sigma, 1)[1] #torch.max函数结果中的索引为1的部分（max会返回两部分数据，索引0对应最大值，1是对应的索引、即标签）
 correct += torch.sum(yhat y) # 对bool类型加和，True为1，False为0
 samples += x.shape[0]

 #每训练一个batch的数据，模型见过的数据就会增加x.shape[0]
 #每N个batch我就打印一次，
 if (batch_idx + 1) % 125 0 or batch_idx len(batchdata)-1:
    print("Epoch{}:[{}/{}({:.0f}%)],Loss:{:.6f},Accuracy:{:.3f}".format(
    epoch+1
    ,samples
    ,epochs*len(batchdata.dataset)
    ,100*samples/(epochs*len(batchdata.dataset))
    ,loss.data.item()#损失值
    ,float(100*correct/samples)
    ))
 #分子代表：已经查看过的数据有多少
 #分母代表：在现有的epochs设置，模型一共需要查看多少数据
```

- 5 进行训练与评估

```python
#实例化神经网络，调用优化算法需要的参数
torch.manual_seed(420)
net = Model(in_features=input_, out_features=output_)
fit(net,batchdata,lr=lr,epochs=epochs,gamma=gamma)
```
