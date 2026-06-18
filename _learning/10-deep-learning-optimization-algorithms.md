---
layout: article
title: "深度学习优化算法"
key: learning-10-deep-learning-optimization-algorithms
permalink: /learning/10-deep-learning-optimization-algorithms.html
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

## 1 梯度不平稳性与Glorot条件

从本节开始，将正式进入到优化方法的具体方法部分内容。首先是关于激活函数使用过程的优化。在上一节的结尾，我们发现，尽管激活函数的使用能够有效提升神经网络模型效果，但激活函数的简单叠加却会让模型出现很多问题。深度学习本来就是深层次神经网络，如果不解决激活函数在叠加过程中遇到的问题，深层次神经网络也就无从构建。

本节将从上一节建模过程中所遇到的、堆叠激活函数之后遇到的问题入手，讨论造成这些梯度爆炸问题的根源，以及如何通过一些优化方法解决这些问题。

- 模型病灶与模型调优

业余算法工程师进行模型调优就像我们平时去药店买药，啥病买啥药，然后挑个贵的；而专业的算法工程师进行模型调优则更像医生为患者进行诊断，在了解病理的基础上，通过一些列更加专业的诊断手段判断病情，然后提出更加高效的综合性解决方案。

因此，初级算法工程师只需要知道模型会有什么样的问题（什么病）以及分别可以用什么方法解决（都有什么药），模型调优时主要靠套用和反复调试（多尝试各种办法），而更加专业的算法工程师则需要**知道这些模型问题背后的成因，也就是到底是什么原因导致了模型出现这些问题（病理），同时也需要掌握更加丰富的诊断模型问题的一些方法**（不仅仅是看模型评估指标结果），然后了解优化方法的基本原理以及发挥作用的方式，当然至关重要的一段是需要在长期实践中不断积累经验，才能最终获得专业性的长足的进步。

正是出于这些角度的考虑，在优化方法的内容中，我们不仅会介绍每一种方法发挥作用的场景，更重要的是我们会同步介绍深度学习模型在建模过程中的一些典型问题的病理、诊断这些问题的角度及方法，以及优化方法背后的“药理”。

### 1.1 梯度消失与梯度爆炸

对于神经网络这个复杂系统来说，在模型训练过程中，一个最基础、同时也最常见的问题，就是梯度消失和梯度爆炸。

我们知道，**神经网络在进行反向传播的过程中，各参数层的梯度计算会涉及到激活函数导函数取值**，具体来说，假设现在有一个三层的神经网络，其中两个隐藏层的激活函数为𝐹(𝑥)，对应的导函数为𝑓(𝑥)，设X为输入训练的数据特征，y为标签，𝑦̂ 为模型向前传播输出结果，𝑤1为第一层参数、𝑤2为第二层参数、𝑤3为第三层参数，loss为损失函数，则有如下计算公式：

每一次正向传播计算结果：
$$
\hat y = F(F(X * w_1) * w_2) * w_3
$$
而loss是一个关于y和𝑦̂ 的函数，而y是常量，𝑦̂ 是一个关于w的函数，因此loss也进行如下表示：
$$
loss(\hat y)
$$
在进行梯度求解时候，假设𝑤1对应梯度为𝑔𝑟𝑎𝑑1,𝑤2对应梯度为𝑔𝑟𝑎𝑑2,𝑤3对应梯度为𝑔𝑟𝑎𝑑3,为了简化计算，我们假设所有的𝑥、𝑤1、𝑤2、𝑤3都是标量，根据链式法则，有计算过程如下：
$$
\begin{aligned}
grad_1 &=\frac{\partial loss}{\partial w_1} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial w_1} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial (F(F(X * w_1) * w_2) * w_3)}{\partial w_1} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial (F(F(X * w_1) * w_2) * w_3)}{\partial (F(F(X * w_1) * w_2)} \cdot \frac{\partial F(F(X * w_1) * w_2)}{\partial F(X * w_1)} \cdot \frac{\partial F(X * w_1)}{\partial w_1}\\
&= \frac{\partial loss}{\partial \hat y} \cdot w_3 \cdot f(F(X * w_1) * w_2) \cdot w_2 \cdot f(X * w_1) \cdot X \\
\end{aligned}
$$
此时𝑔𝑟𝑎𝑑1中计算了两次激活函数的导函数，并且在上述过程中，𝑋∗𝑤1是第一层隐藏层接收到的数据，而𝐹(𝑋∗𝑤1)∗𝑤2则是第二层隐藏层接收到的数据。而对比如果是计算𝑤2的梯度，则有如下过程：
$$
\begin{aligned}
grad_2 &=\frac{\partial loss}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial (F(F(X * w_1) * w_2) * w_3)}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial (F(F(X * w_1) * w_2) * w_3)}{\partial (F(F(X * w_1) * w_2)} \cdot \frac{\partial F(F(X * w_1) * w_2)}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y}
\cdot
w_3 \cdot \frac{\partial F(F(X * w_1) * w_2)}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y}
\cdot
w_3 \cdot f(F(X*w_1)*w_2) \cdot
\frac{\partial (F(X * w_1) * w_2)}{\partial w_2} \\
&= \frac{\partial loss}{\partial \hat y}
\cdot
w_3 \cdot f(F(X*w_1)*w_2) \cdot F(X * w_1) \\
\end{aligned}
$$
我们发现，在计算过程中只出现了一次激活函数的导函数。当然如果我们是计算𝑤3的梯度，则与如下计算过程：
$$
\begin{aligned}
grad_3 &=\frac{\partial loss}{\partial w_3} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial w_3} \\
&= \frac{\partial loss}{\partial \hat y} \cdot \frac{\partial (F(F(X * w_1) * w_2) * w_3)}{\partial w_3} \\
&= \frac{\partial loss}{\partial \hat y} \cdot F(F(X * w_1) * w_2) \\
\end{aligned}
$$
此时𝑔𝑟𝑎𝑑3在计算过程中就已经不涉及激活函数的导函数的计算了

其实如果当**神经网络层数继续增加、激活函数的数量继续增加**，第一层参数在计算梯度的过程中需要相乘的激活函数导函数个数也会随之增加，而后面几层参数的梯度计算中涉及到的激活函数导函数个数逐级递减。

- **梯度爆炸**：

而累乘就容易造成指数级变化，当**激活函数值𝐹(𝐹(𝑋∗𝑤1))、激活函数导函数值𝑓(𝑋∗𝑤1)或者参与相乘的参数取值（𝑤3）较大(>1)时**，会出现𝑔𝑟𝑎𝑑1远大于𝑔𝑟𝑎𝑑2远大于𝑔𝑟𝑎𝑑3的情况，也就是神经网络前几层参数梯度非常大、而后几层参数梯度相对较小的情况，此时就被称为**梯度爆炸**，并且受到**累乘效应的影响**，**前几层梯度也会大于甚至远大于1**，此时就会造成模型迭代过程不稳定的情况发生；

- **梯度消失**

而反之如果**上述几个变量均小于1，甚至远小于1**，则会出现前几层参数梯度非常小、而后几层参数梯度非常大的情况，此时就被称为**梯度消失**，此时由于模型各层参数学习率伴随层数增加逐渐增加，并且由于构成梯度的基本参数均小于1，因此最后几层梯度也会小于1甚至远小于1，此时前几层参数梯度取值将非常小，甚至趋于0，因而会使得前几层的参数无法在迭代中得到更新。

接下来，我们就从梯度消失和梯度爆炸的角度剖析Sigmoid和tanh激活函数叠加过程中可能存在的隐患。

## 2 Sigmoid和tanh激活函数的梯度更新问题

### 2.1 Sigmoid和tanh激活函数的梯度更新问题

#### 2.1.1 理论

对于sigmoid激活函数来说，简答的叠加是极容易出现梯度消失的问题。sigmoid函数及导函数图像如下所示：

```python
# 绘制Sigmoid函数的函数图像和导函数图像
X = torch.arange(-5, 5, 0.1)
X.requires_grad=True
sigmoid_y = torch.sigmoid(X)
```

sigmoid_y.sum()：因为反向传播针对**标量**，一维变量则求导，多维则求梯度（偏导）。

```python
# 反向传播
sigmoid_y.sum().backward()
```

```python
# Sigmoid函数图像
plt.subplot(121)
plt.plot(X.detach(), sigmoid_y.detach())
plt.title("Sigmoid Function")
# Sigmoid导函数图像
plt.subplot(122)
plt.plot(X.detach(), X.grad.detach())
plt.title("Sigmoid Derivative function")
```

```python
plt.plot(X.detach(), X.grad.detach())
plt.title("Sigmoid Derivative function")
```

![1691539756248]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_002.png' | relative_url }})

可以看出，Sigmoid导函数最大值为0.25（在0点处取到），**当x较大或者较小时，导函数取值趋于0**。

梯度消失或者梯度爆炸，始终是个概率问题。我们不能说导函数取值取值小就一定会发生梯度消失问题，只是导函数最大值越小，越有可能发生梯度消失

#### 2.1.2 Sigmoid函数饱和区间

一般来说我们会将**靠近sigmoid函数的左右两端的区间**称为函数的**饱和区间**（如下图圈出部分）（也就是自变量绝对值较大的区间），不难发现，当自变量落入饱和区间时，因变量会趋于0或者1，而无论自变量是极小（负数绝对值极大）还是极大，都会使得导函数取值趋于0，从而更容易导致模型梯度消失。

#### 2.1.3 Sigmoid激活函数叠加后的梯度消失问题

- **模型参数及梯度提取方法**

由于需要对各线性层参数的变化情况进行观察，此处我们也简单总结此前提到的关于模型参数和及其对应梯度的提取方法。

```python
# 创建随机数种子
torch.manual_seed(420)

# 实例化模型
sigmoid_model3 = Sigmoid_class3()
```

1）首先是利用.parameters()方法观察模型所有参数。

使用**list将其转化为显式的列表**。此时列表中的每个元素都是带名称的张量 。可用索引获取每一个w或截距b

```python
pl = list(sigmoid_model3.parameters())
pl
'''[Parameter containing:
 tensor([[ 0.4318, -0.4256],
 [ 0.6730, -0.5617],
 [-0.2157, -0.4873],
 [ 0.5453, 0.2653]], requires_grad=True),
 Parameter containing:
 tensor([-0.3527, -0.5469, -0.4094, -0.1364], requires_grad=True),'''

pl[0]
'''Parameter containing:
tensor([-0.3527, -0.5469, -0.4094, -0.1364], requires_grad=True)'''
```

.parameters()方法返回模型所有参数，包括截距，但需要使用list将其转化为显式的列表。此时列表中的每个元素都是带名称的张量，关于带名称的张量会在此后课程中介绍，此处我们只需要知道**模型参数本身也是张量**，并且是可微张量即可。并且，就像此前说到的一样，**推导**过程中我们使用数据是按照行排列，**一行代表一条数据，一列代表数据的一个特征**，而**PyTorch 在进行计算时，会将数据转化为按列排列，一列代表一条数据**，因此 PyTorch 中我们查看到的**模型参数和推导的模型参数互为转置关系**。

2） 通过.modules方法，观察构成该模型的每个线性层的(genareator)参数。 ml中，**除了第一个元素是自定义的类以外**，其他的都是构成模型的**线性层**。

```python
ml = list(sigmoid_model3.modules())
ml
'''[Sigmoid_class3(
 (linear1): Linear(in_features=2, out_features=4, bias=True)
 (linear2): Linear(in_features=4, out_features=4, bias=True)
 (linear3): Linear(in_features=4, out_features=4, bias=True)
 (linear4): Linear(in_features=4, out_features=1, bias=True)
 ),
 Linear(in_features=2, out_features=4, bias=True),
 Linear(in_features=4, out_features=4, bias=True),
 Linear(in_features=4, out_features=4, bias=True),
 Linear(in_features=4, out_features=1, bias=True)]'''
```

- 返回模型第一层输出结果

```python
f = torch.tensor([[1., 2]]) #一条样本数据
f
sigmoid_model3(f) # 返回最终输出结果
ml = list(sigmoid_model3.modules())
ml[1](f) # 样本数据为f时，返回模型第一层输出结果
'''tensor([[-0.7721, -0.9973, -1.5997, 0.9395]], grad_fn=<AddmmBackward0>)'''
```

- 线性层可以使用weight返回其自变量权重，通过bias返回其截距 :

**方法1**：（通过.data获取数据）

```python
ml[1].weight #pl[0]
'''
Parameter containing:
tensor([[ 0.4318, -0.4256],
 [ 0.6730, -0.5617],
 [-0.2157, -0.4873],
 [ 0.5453, 0.2653]], requires_grad=True'''

ml[1].weight.data
'''tensor([[ 0.4318, -0.4256],
 [ 0.6730, -0.5617],
 [-0.2157, -0.4873],
 [ 0.5453, 0.2653]])'''

ml[1].bias
'''Parameter containing:
tensor([-0.3527, -0.5469, -0.4094, -0.1364], requires_grad=True)'''
```

**方法2**：

```python
# 获取weight张量对象
sigmoid_model3.linear1.weight
'''Parameter containing:
tensor([-0.3527, -0.5469, -0.4094, -0.1364], requires_grad=True)'''
# 获取bias张量对象

sigmoid_model3.linear1.weight.data
'''tensor([[ 0.4318, -0.4256],
 [ 0.6730, -0.5617],
 [-0.2157, -0.4873],
 [ 0.5453, 0.2653]])'''

sigmoid_model3.linear1.bias
'''Parameter containing:
tensor([-0.3527, -0.5469, -0.4094, -0.1364], requires_grad=True)'''
```

```python
# 提取weight
for m in sigmoid_model3.modules():
    if isinstance(m, nn.Linear):
        print(m.weight)

# 提取bais
for m in sigmoid_model3.modules():
    if isinstance(m, nn.Linear):
        print(m.bias)
```

- 当然，我们也可直接使用.grad查看梯度

**方法1**：

```python
# 提取weight梯度
for m in sigmoid_model3.modules():
 if isinstance(m, nn.Linear):
 print(m.weight.grad)
```

**方法2**：也可以获取到每一层，再去获得grad

```python
# 获取weight张量梯度
sigmoid_model3.linear1.weight.grad
```

可以通过torch.nn.Parameter实现模型所有层的权重与偏置的初始化：

torch.nn.parameter.Parameter(data=None, requires_grad=True)

torch.nn.Parameter是继承自torch.Tensor 的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor 的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
注意到，nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torth.Tensor 对象的默认值相反。

在nn.Module类中，PyTorch 也是使用nn.Parameter来对每一个module的参数进行初始化的。

关于训练模型时是每一轮开始时清零梯度还是结束时清零梯度，本质上效果相同，由于此处我们需要查看每一轮迭代完的梯度，因此此前的函数创建时采用每轮开始时清除上一轮梯度的设置。

```python
fit(net = sigmoid_model3,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD(sigmoid_model3.parameters(), lr = 0.03),
 batchdata = train_loader,
 epochs=5,
 cla=False)

for m in sigmoid_model3.modules():
    if isinstance(m, nn.Linear):
    print(m.weight.grad)
'''tensor([[-0.0043, -0.0027],
 [-0.0176, -0.0117],
 [-0.0040, -0.0021],
 [ 0.0098, 0.0052]])
tensor([[ 0.0062, 0.0077, -0.0007, 0.0115],
 [-0.0107, -0.0133, 0.0014, -0.0202],
 [-0.0341, -0.0421, 0.0027, -0.0627],
 [ 0.0124, 0.0154, -0.0012, 0.0229]])
tensor([[ 0.0066, 0.0060, 0.0113, 0.0083],
 [ 0.0209, 0.0191, 0.0359, 0.0265],
 [-0.0773, -0.0708, -0.1329, -0.0979],
 [-0.1050, -0.0960, -0.1804, -0.1329]])
tensor([[-0.6299, -0.7897, -0.8889, -0.7292]])'''
```

从上述结果能够看出前几层梯度较小，后几层梯度较大。当然，一种更加直观的观测手段，是通过绘制小提琴图来对各层梯度进行观察，具体过程如下：

```python
import seaborn as sns # 导入seaborn绘图包，后续需要将其添加进启动项
```

```python
vp = [] # 创建空列表用于存储每一层参数的梯度
```

```python
for i, m in enumerate(sigmoid_model3.modules()):
    if isinstance(m, nn.Linear):
        vp_x = m.weight.grad.detach().reshape(-1, 1).numpy() # 每一层参数梯度
        vp_y = np.full_like(vp_x, i) # 对层进行标记
        vp_a = np.concatenate((vp_x, vp_y), 1)
        vp.append(vp_a)
        '''[array([[-0.00434157, 1. ],
 [-0.00269753, 1. ],
 [-0.01757523, 1. ],
 [-0.01171168, 1. ],
 [-0.00395311, 1. ],
 [-0.00208129, 1. ],
 [ 0.00981918, 1. ],
 [ 0.0052211 , 1. ]], dtype=float32),
 array([[ 6.23247027e-03, 2.00000000e+00],
 [ 7.73869036e-03, 2.00000000e+00],
 [-6.50476199e-04, 2.00000000e+00],
 [ 1.15341265e-02, 2.00000000e+00],
 [-1.06804930e-02, 2.00000000e+00],
 [-1.33101307e-02, 2.00000000e+00],
 [ 1.41562056e-03, 2.00000000e+00],
 [-2.02342980e-02, 2.00000000e+00],
 [-3.41279954e-02, 2.00000000e+00],
 [-4.21172231e-02, 2.00000000e+00],
 [ 2.70337984e-03, 2.00000000e+00],
 [-6.27222657e-02, 2.00000000e+00],
 [ 1.24284420e-02, 2.00000000e+00],
 [ 1.54135972e-02, 2.00000000e+00],
 [-1.21727260e-03, 2.00000000e+00],
 [ 2.29195952e-02, 2.00000000e+00]], dtype=float32),
 array([[ 0.00657503, 3. ],
 [ 0.00601594, 3. ],
 [ 0.01130071, 3. ],
 [ 0.00832438, 3. ],'''
```

```python
vp_r = np.concatenate((vp), 0) # 拼接行
vp_r
```

绘图：

```python
ax = sns.violinplot(y = vp_r[:, 0], x = vp_r[:, 1])
ax.set(xlabel='num_hidden', title='Gradients')
```

![1691552926627]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_003.png' | relative_url }})

小提琴图其实是一种统计分析图像，基本含义如下所示：

![1691552954581]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_004.png' | relative_url }})

对于复杂模型来说，如果部分层失去学习能力（参数迭代的可能性，梯度很小，参数变化小），其实模型判别效力就和简单模型无异。从此也能看出对复杂模型进行有效训练的重要性。

## 3 tanh函数的梯度计算问题

如果说Sigmoid激活函数叠加更容易造成梯度消失，那tanh激活函数叠加是不是也是如此呢？首先我们来观察tanh激活函数导函数性质。

```python

# 绘制tanh函数的函数图像和导函数图像
X = torch.arange(-5, 5, 0.1)
X.requires_grad=True
tanh_y = torch.tanh(X)

# 反向传播
tanh_y.sum().backward()

# tanh函数图像
plt.subplot(121)
plt.plot(X.detach(), tanh_y.detach())
plt.title("tanh Function")

# tanh导函数图像
plt.subplot(122)
plt.plot(X.detach(), X.grad.detach())
plt.title("tanh Derivative function")
```

![1691561207154]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_005.png' | relative_url }})

对于tanh函数来说，导函数的取值分布在0-1之间的，看似导函数取值累乘之后也是趋于0的，但实际上，tanh激活函数的叠加即可能造成梯度消失、同时也可能造成梯度爆炸，原因是在实际建模过程中，影响前几层梯度的其他变量大多数情况都大于1，因此对于一个导函数极大值可以取到1的激活函数来说，还是有可能出现梯度爆炸的情况的（只要权重大于1）。

![1691562621974]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_006.png' | relative_url }})

随着迭代次数增加，这种梯度爆炸的情况有增无减。

![1691562633393]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_007.png' | relative_url }})

从根本上来说，tanh激活函数的迭代不平稳就是因为部分层的部分梯度存在极端值，当然，这种极端值也导致部分层无法有效学习、最终影响模型效果。

## 4 Zero-Centered Data与Glorot条件

通过对Sigmoid和tanh激活函数叠加后的模型梯度变化情况分析，我们不难发现，梯度不平稳是影响模型建模效果的非常核心的因素。而这个看似简单问题的解决方案，却花费了研究人员数十年的时间才逐渐完善，我们现在所接触到的优化方法，也基本上是在15年前后提出的，而这些被验证的切实可行的优化方法，也是推动这一轮深度学习浪潮的技术因素。

当然，这些优化方法主要是针对深层次神经网络的。

整体来看，针对**梯度不平稳的解决方案**（优化方法）总共分为五类，分别是1）参数初始化方法、2）输入数据的归一化方法、3）衍生激活函数使用方法、4）学习率调度方法以及5）梯度下降优化方法。接下来，先介绍所有上述优化算法的一个基本理论，由Xavier Glorot在2010提出的Glorot条件。

### 4.1 Zero-centered Data

在介绍Glorot条件之前，我们先从一个更加朴素的角度出发，讨论关于Zero-Centered Data相关作用，从而帮助我们理解后续Glorot条件。

首先，我们还是假设当前模型是一个三层神经网络，其中两个隐藏层的激活函数为𝐹(𝑥)，对应的导函数为𝑓(𝑥)，设X为输入训练的数据特征，y为标签，𝑦̂ 为模型向前传播输出结果，𝑤1为第一层参数、𝑤2为第二层参数、𝑤3为第三层参数，loss为损失函数，则有如下计算公式：

每一次正向传播计算结果：
$$
\hat y = F(F(X * w_1) * w_2) * w_3
$$
依次类推。而在反向传播过程，各参数层的梯度如下 :
$$
\begin{aligned}grad_1&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot w_2\cdot f(X*w_1)\cdot X\\\\grad_2&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot F(X*w_1)\\\\grad_3&=\frac{\partial loss}{\partial\hat{y}}\cdot F(F(X*w_1)*w_2)\end{aligned}
$$

在梯度消失和梯度爆炸的案例中，我们不难发现，为了确保多层神经网络的有效性，各层梯度的差距不应太大，此时一个最为基本的想法就是，就是能否让所有的输入数据（也就是X）以及所有层的参数都设置为**Zero-Centered Data**（零均值随机数），也就是**零点对称数据**，不难发现，由于X和𝑤𝑖都是零点对称的，**因此每一个线性层中的导函数也取值也能够维持在0-1之间（对于tanh，零均值数据会让f(F())，F（）在0附近，导数为1附近，则会消除这个累乘的影响），进而每一层的梯度基本能维持在比较平稳的状态。**

所以如何将X和𝑤𝑖都转为**Zero-Centered Data**是核心问题：

很明显，我们不能将参数的初始值全部设为0，我们只能考虑借助统计工具生成均值是0的随机数，也就是0均值的均匀分布或者是0均值的高斯分布，但这里需要考虑的另一个问题就是，**该随机数的方差应该如何确定**？

### 4.2 Glorot条件和Xavier方法

初始化参数的方差如何确定这一问题在一个严谨论述如何保证模型有效性的论文中，从另一个角度出发，得到了回答。根据Xavier Glorot在2010年发表的[《Understanding the difficulty of training deep feedforward neural networks》](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)论文中的观点 ：

为保证模型本身的有效性和稳定性，我们希望正向传播时，每个线性层输入数据的方差等于输出数据的方差，同时我们也希望反向传播时，数据流经某层之前和流经某层之后该层的梯度也具有相同的方差，虽然二者很难同时满足（除非相邻两层神经元个数相同），但Glorot和Bengio（论文第二作者）表示，如果我们适当修改计算过程、是可以找到一种折中方案去设计初始参数取值，从而同时保证二者条件尽可能得到满足，这种设计参数初始值的方法也被称为Xavier方法，而这种方法也经过一段时间的实践验证被证明是很好的一种初始化模型参数的方法，尤其是对于使用tanh激活函数的神经网络来说，效果更为显著。

### 4.3 模型初始化参数取值影响

模型初始化时得到的不同参数，本质上等价于在损失函数上找到了不同的初始点，而同一损失函数，初始点选取的不同应该不会影响最终迭代结果才对，但事实情况并非如此。

在模型实例化过程中，**采用不同随机数种子**，就相当于选取了不同的模型初始参数。

初始参数值的选取不仅会影响模型收敛速度，甚至在某些情况下还会影响模型的最终表现。造成此现象的根本原因还是在于神经网络模型在进行训练时，不确定性过多，而在一个拥有诸多不确定性的系统中再加上不确定的初始参数，初始参数的不确定性会被这个系统放大。并且，值得一提的是，每一个epoch中的每一次迭代并不是在一个损失函数上一步步下降的，当我们使用小批量梯度下降算法时，带入不同批的数据，实际创建的损失函数也会不同。

## 5 Dead ReLU Problem与学习率优化

和Sigmoid、tanh激活函数不同，ReLU激活函数的叠加并不会出现梯度消失或者梯度爆炸，但ReLU激活函数中使得部分数值归零的特性却会导致另外一个严重的问——Dead ReLU Problem，也被称为神经元活性失效问题。

### 5.1 Dead ReLU Problem成因分析

#### 5.1.1 Dead ReLU Problem直接表现

![1691563258060]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_008.png' | relative_url }})

模型在迭代多轮之后，训练误差和测试误差都在各自取值的高位收敛了，也就是误差不随模型迭代测试增加而递减。通过简单尝试我们不难发现，此时模型对所有数据的输出结果都是0。

```python
relu_model3(features)
'''tensor([[0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [0.],'''
```

### 5.2 Dead ReLU Problem成因分析

#### 5.2.1 Dead ReLU Problem基本判别

神经元活性失效问题和ReLU激活函数本身特性有关。

```python
 绘制ReLU函数的函数图像和导函数图像
X = torch.arange(-5, 5, 0.1)
X.requires_grad=True
relu_y = torch.relu(X)
# 反向传播
relu_y.sum().backward()

# ReLU函数图像
plt.subplot(121)
plt.plot(X.detach(), relu_y.detach())
plt.title("ReLU Function")
# ReLU导函数图像
plt.subplot(122)
plt.plot(X.detach(), X.grad.detach())
plt.title("ReLU Derivative function")
```

![1691563507209]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_009.png' | relative_url }})

对于ReLU激活函数来说，只要激活函数接收到的数据小于0，输出结果就全是0，并且更关键的是，只要ReLU输出结果是0，**由于ReLU的导函数是分段常数函数且接收数据为负时导数为0，因此如果ReLU输出结果为零**，则反向传播结果、也就是各层的梯度，也都是零。

我们发现，当某条数据在模型中的输出结果为0时，反向传播后各层参数的梯度也全为0，此时参数将无法通过迭代更新(**损失趋于平稳**)。

而进一步的，如果在某种参数情况下，整个训练数据集输入模型之后输出结果都是0，则在小批量梯度下降的情况下，每次再挑选出一些数据继续进行迭代，仍然无法改变输出结果是0的情况，此时**参数无法得到更新、进而下次输入的小批数据结果还是零、从而梯度为0、从而参数无法更新.**..至此陷入死循环，模型失效、激活函数失去活性，也就出现了Dead ReLU Problem。

![1691564557953]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_010.png' | relative_url }})

当然，上述过程可以由如下数学过程说明，假设模型预测值:
$$
\hat y = ReLU(X * w_1) * w_2
$$
并且，出现Dead ReLU Problem的时候，某一组𝑤1恰好使得ReLU输出结果为0，因此𝑦̂ =0，而此时损失函数为：
$$
loss = MSE = \frac{\sum ((\hat y - y) ^ 2)}{N}
$$
根据链式法则，此时梯度为：
$$
\begin{aligned}
grad &= ( \frac{\partial loss}{\partial w_1}, \frac{\partial loss}{\partial w_2}) \\
&= (\frac{\partial loss}{\partial \hat y} \frac{\partial \hat y}{\partial w_1}, \frac{\partial loss}{\partial \hat y} \frac{\partial \hat y}{\partial w_2})\\
\end{aligned}
$$
由于𝑦̂ 恒为0（为一个常量），由ReLU激活函数导函数特性可知，𝑦̂ 对任何自变量偏导也为0，此时grad也就为0，𝑤1、𝑤2不会再更新。而参数不更新，𝑦̂ 仍然为0，梯度还是为0，如此往复，就出现了上述死循环，也就是Dead ReLU Problem。

#### 5.2.2 Dead ReLU Problem发生概率

如果是两层ReLU层，模型结构如下：

![1691564800632]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_001.png' | relative_url }})

向前传播过程中，模型输出结果为：
$$
\hat y = ReLU(ReLU(X * w_1) * w_2) * w_3
$$
在两层ReLU嵌套的情况下，𝑦̂ 取值为0的概率就更大了，出现Dead ReLU Problem的概率也就更高了。而同时，根据**各层参数的梯度计算公式我们也能够发现，只要其中任意一层输出结果是0，则所有层参数的梯度均为0。**
$$
\begin{aligned}grad_1&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot w_2\cdot f(X*w_1)\cdot X\\\\grad_2&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot F(X*w_1)\\\\grad_3&=\frac{\partial loss}{\partial\hat{y}}\cdot F(F(X*w_1)*w_2)\end{aligned}
$$
当然，如果模型是带入偏差进行的建模，**出现Dead ReLU Problem的时候模型输出结果恒为bias的取值**。

### 5.3 通过调整学习率缓解Dead ReLU Problem

在所有的解决Dead ReLU Problem的方法中，最简单的一种方法就是调整学习率。尽管我们知道，ReLU叠加越多层越容易出现神经元活性失效，但我们可以简单通过降低学习率的方法来缓解神经元活性失效的问题。甚至可以说这是一种通用且有效的方法。

此前我们曾介绍学习率越小、收敛速度就越慢，而学习率过大、则又容易跳过最小值点造成模型结果震荡。对于ReLU激活函数来说，参数“稍有不慎”就容易落入输出值全为0的陷阱，因此训练过程需要更加保守，采用更小的学习率逐步迭代。当然学习率减少就必然需要增加迭代次数，但由于ReLU激活函数计算过程相对简单，增加迭代次数并不会显著增加计算量。

### 5.4 ReLU激活函数特性理解

ReLU激活函数的实际作用是“选择性更新部分参数”，回顾上述公式，我们不难发现，当激活函数是ReLU激活函数时，以上述例子为例，无论是第一层接收到的数值（𝑋∗𝑤1）小于0，还是第二层接收到的数值（𝐹(𝑋∗𝑤1)∗𝑤2小于0，都会导致三层参数的梯度为0，同时，哪怕是第一层接收到的数据全是0，也会导致所有参数本轮不被更新。而在实际深度学习建模过程中，我们是采用小批量梯度下降算法来进行梯度计算，而如果某一批的数据输出结果为0，则当前迭代结束时参数不变，也就**相当于模型采用了“有选择”的数据进行参数更新——只“选择”了那些输出结果不为0的数据进行参数训练。同时我们需要知道的是，这一轮某一批数据没被选择，不代表下一轮这批数据仍然不被选择（因为参数会变化，因而输出结果也会发生变化），因此我们可以理解为每一轮都带入不同数据进行参数训练，从而最终完成模型训练。**

并且，从梯度消失和梯度爆炸角度考虑，ReLU激活函数拥有更加优异的特性。我们都知道，**ReLU激活函数的导函数取值要么是1要么是0**，当**导函数取值为1时，能够保证各层参数的梯度在计算时不受因层数变化而累乘的导函数影响**（因为导函数取值都为1）。

- 值得注意的是，每次带入不同批次数据训练，或者说每次有选择性的忽略部分数据，就相当于进行了“非线性”的变换。

关于“随机性”的作用，其实我们已经见过很多次了，

- 在集成模型中，我们将在一定随机性条件下构建的、彼此不同的基分类器进行集成，从而创建一个效果明显好于单个基分类器的集成模型；
- 在SGD中，我们采用每次带入**随机部分数据**的方法进行梯度下降迭代，从而使得迭代过程能够跳出局部最小值点；
- 而在ReLU激活函数中，我们随机**挑选部分**参数进行迭代，从而完成数据的“**非线性**”转化，进而保证模型本身的有效性。

也正因如此，**ReLU激活函数**也被称为非饱和激活函数

## 6 nn.Sequential快速建模方法及nn.init 模型参数 自定义方法

在讨论如何解决上述激活函数叠加问题之前，我们先补充两个基础工具，其一是使用nn.Sequential进行模型的快速构建，其二则是使用nn.init来进行模型参数修改。其中关于模型参数修改的相关方法，也是支撑本节优化方法实践的核心。

### 6.1 nn.Sequential快速建模方法介绍（只能是线性结构）

在此前的建模环节，我们都是通过完整的创建模型类、通过定义init方法和forward方法来确定模型的基本结构、传播方式和激活函数。除了这种模型定义方法外，PyTorch 还支持使用nn.Sequential来快速，在借助nn.Sequential进行模型构建过程中，我们只需要将每一层神经元连接方法和激活函数作为参数输入nn.Sequential即可，具体流程如下：

两层全连接神经网络，并且隐藏层使用ReLU进行处理。其中需要注意的是，nn.ReLU()单独使用时就相当于ReLU函数，而放在nn.Sequential中就等价于对某一层的输出结果进行ReLU处理。

```python
# 设置随机数种子
torch.manual_seed(25)

# 构建上述LR_ReLU_test模型
relu_test = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
```

relu_test相当于已经实例化之后的模型。

而此时的实例化，是nn.Sequential类的实例化，也就是说，通过nn.Sequential创建的模型本质上都是nn.Sequential的一个实例。

```python
isinstance(relu_test, nn.Sequential)
# True
```

### 6.2 模型参数自定义方法

继续补充关于手动设置模型初始参数及模型参数共享的方法。首先，对于模型参数来说，parameters返回结果是个生成器（generator），通过list转化后会生成一个由可微张量构成的list。

```python
# 重新设置初始化模型参数取值
# 设置随机数种子
torch.manual_seed(25)

# 构建上述LR_ReLU_test模型
relu_test = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
```

1）通过修改可微张量方法修改参数。对于创建的x为行为参数的张量，需要转置，才能赋值，因为.parameters()中是建立的转置。

```python
# 修改目标
w1 = torch.tensor([[0., 0], [-1, -2]])
w2 = torch.tensor([1., -1]).reshape(-1, 1)

list(relu_test.parameters())[0].data = w1.t()
```

2）使用init方法创建满足某种分布的参数

除了通过手动方法修改参数值以外，我们还可以使用nn.init方法来对模型参数进行修改。

- **.nn.init.uniform_(张量，均匀分左边界，均匀分布右边界)**方法，新生成的参数服从均匀分布

’_‘是将原张量修改为经过函数后的张量

```python
nn.init.uniform_(list(relu_test.parameters())[0], 0, 1) # 设置参数值为均匀分布在0，1区间内的随机数
```

- .nn.init.normal_(张量，均值，方差)方法，新生成的参数服从正态分布

```python
nn.init.normal_(list(relu_test.parameters())[0], 0, 1) # 服从均值为0、标准差为1的正态分布
```

- .nn.init.constant_方法，新生成的参数值为某一常数

```python
nn.init.constant_(list(relu_test.parameters())[0], 1) # 参数全为1
```

## 7 Xavier方法与kaiming方法（HE初始化）参数初始化方法

**只是对初始参数做到零均值数据，以后的每次迭代则会按照梯度修改。数学归一化也是只在输入X上做一次归一化。**

### 7. 1 Xavier初始化参数方法基础理论

回顾Glorot条件，我们要求**正向传播时候数据流经每一层（线性层）前后的方差一致，并且反向传播时候数据流经每一层（线性层），该层梯度前后方差一致**。我们将前者称为向前传播条件，后者称为反向传播条件。

- 向前传播条件：**正向传播时候数据流经每一层（线性层）前后的方差一致**

- 反向传播条件：**反向传播时候数据流经每一层（线性层），该层梯度前后方差一致**

我们先看当向前传播条件成立时候，有如下计算过程

首先，数据流至某一层时，该层神经元接收到的数据可以通过如下方法算得：
$$
z_j = \sum^n_{i=1}w_ix_i
$$
其中𝑧𝑗表示当前某层神经元接收到的数据，𝑥𝑖表示上一层某神经元传出的数据，𝑤𝑖代表连接对应两个神经元之间的权重，当然，如果我们将𝑧𝑗、𝑥𝑖、𝑤𝑖看成是随机变量，则上式就能够表示计算过程的一般情况。并且𝑤𝑖和𝑥𝑖的方差计算过程如下：
$$
Var(w_ix_i) = E[w_i]^2Var(x_i)+E[x_i]^2Var(w_i)+Var(w_i)Var(x_i)
$$

上述过程涉及基础数学知识，设X为随机变量，$D(X)$表示X方差，$E(X)$表示其期望，则有：
$$
D(X)=E(X^2)-E(X)^2
$$
进一步，如果X、Y表示随机变量并且相互独立，则有：
$$
D(XY)=E((XY)^2)-(E(XY))^2=E(X^2)E(Y^2)-(E(X)E(Y))^2
$$
并且对其进行简单变形，可以得出：
$$
\begin{aligned}
D(XY) &=E(X)^2D(Y)+E(Y)^2D(X)+D(X)D(Y) \\
&= E(X)^2[E(Y^2)-E(Y)^2]+E(Y)^2[E(X^2)-E(X)^2]+[E(X^2)-E(X)^2][E(Y^2)-E(Y)^2] \\
&= E(X)^2E(Y^2)-E(X)^2E(Y)^2+E(Y)^2E(X^2)-E(Y)^2E(X)^2+E(X^2)E(Y^2)-E(X)^2E(Y^2)-E(X^2)E(Y)^2+E(X)^2E(Y)^2 \\
&= E(X^2)E(Y^2)-E(X)^2E(Y)^2\\
\end{aligned}
$$
其中Var()表示方差计算，E()表示均值计算。由于我们假设参数是以0为均值的均匀分布或者0为均值的正态分布，因此𝐸(𝑤𝑖)=0，而此前我们介绍，假设输入数据是Zero-Centered的，因此𝐸(𝑥𝑖)=0，所以上式可进一步简化为：
$$
Var(w_ix_i) = Var(w_i)Var(x_i)
$$
而对于z来说，$z = \sum^n_{i=1}w_ix_i$，其方差计算过程如下：
$$
Var(z) = \sum^n_{i=1} Var(w_ix_i) = \sum^n_{i=1} Var(w_i)Var(x_i)
$$
此处补充数学过程，$$D(X+Y)=D(X)+D(Y)+COV(X,Y)$$
其中$COV(X,Y)$是二者的协方差，当$X、Y$相互独立时，$COV(X,Y)=0$。

此时，我们可以认为$x_{(1-n)}和w_{(1-n)}$都是独立同分布的（一个是采集处理后的数据，一个是随机生成的参数， n是上一层神经元的个数 ），因此每个𝑉𝑎𝑟(𝑤𝑖)𝑉𝑎𝑟(𝑥𝑖)也是独立同分布的，因此所有的𝑤𝑖都可以用一个随机变量𝑤表示，所有的𝑥𝑖也可以用一个随机变量𝑥表示，上式可进一步简写为：
$$
Var(z) = \sum^n_{i=1} Var(w_i)Var(x_i) = nVar(w)Var(x)
$$
而我们希望线性变换不影响数据方差（向前、反向传播条件），因此有：
$$
Var(z)=Var(x)
$$
进一步可以算得：
$$
Var(w) = \frac{1}{n}
$$
其中，n是上一层神经元的个数。需要注意的是，上式只考虑了正向传播的情况，而实际在进行反向传播时候，上述过程正好相反。反向传播时z代表上一层神经元接收到的数据，而x则代表当前层传出的数据，虽然计算公式不变，但n的含义却发生了变化。为了进行区分，我们将正向传播时的n、也就是代表**上一层神经元个数的变量**，命名为$𝑛_{𝑖𝑛}$，而在进行反向传播时的n、也就是代表**当前层神经元个数的变量**，命名为$𝑛_{𝑜𝑢𝑡}$。为了同时兼顾向前传播和反向传播这两种情况，我们将w的最终**方差**取值为:
$$
Var(w) = \frac{1}{\frac{n_{in}+n_{out}}{2}} = \frac{2}{n_{in}+n_{out}}
$$
取为均值，也就是折中的方案 。

而此时，如果我们设置w服从均匀在[-a, a]区间内均匀分布，则w的方差为：
$$
Var(w) = \frac{(a+a)^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3} = \frac{2}{n_{in}+n_{out}}
$$
注：服从在[a, b]区间上均匀分布的随机变量方差为$\frac{(b-a)^2}{12}$

进而计算可得 ：
$$
a = \sqrt{\frac{6}{n_{in}+n_{out}}}
$$
即w的是均匀分布在$(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}})$上的随机变量。

当然，如果我们假设w是服从正态分布的，则w服从$(0, \sqrt{\frac{2}{n_{in}+n_{out}}})$的随机变量。据此，我们就能设置每一层的初始参数了。当然，在 PyTorch 中，可以使用init方法进行初始化。

> 另外，一种更加严谨的、指代某一次传播过程上一层神经元数量和下一层神经元数量的叫法是扇入(fan in)和扇出(fan out)，由此Xavier方法中初始参数的方差也可写为$Var(w) = \frac{2}{fan_{in}+fan_{out}}$。另外，Xavier在论文中所指出的，应该保持各层的激活值和梯度的方差在传播过程中保持一致，也被称为Glorot条件。

### 7.2 Xavier初始化参数执行方法

#### 7.2.1 PyTorch实现Xavier均匀分布的参数创建

1）我们可以使用`torch.nn.init.xavier_uniform_`(tensor, gain=1.0)进行初始化参数均匀分布设置。

当然，这里需要注意的是，函数处理的对象是张量，也就是说只要输入张量，就可以自动进行处理。其中$fan_{in}$由张量的列决定，$fan_{out}$由张量的行数决定。（想想是为什么？）

从说明中我们能看到，此时均匀分布的边界为
$$
a = \text{gain} \times \sqrt{\frac{6}{fan_{in} + fan_{out}}}
$$
其中gain为增益系数，默认为1，用于手动调整均匀分布的边界，暂时不建议调整。

- Sigmoid激活函数

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建最高项为2的多项式回归数据集
features, labels = tensorGenReg(w=[2, -1], bias=False, deg=2)

# 进行数据集切分与加载
train_loader, test_loader = split_loader(features, labels)

# 初始核心参数
lr = 0.03
num_epochs = 20

```

```python
# 设置随机数种子
torch.manual_seed(420)

# 实例化模型
sigmoid_model3 = Sigmoid_class3() # 保留原参数
sigmoid_model3_init = Sigmoid_class3() # 使用Xavier初始化参数

# 修改init模型初始参数
for m in sigmoid_model3_init.modules():
 if isinstance(m, nn.Linear):
 nn.init.xavier_uniform_(m.weight)

# 创建模型容器
model_l = [sigmoid_model3, sigmoid_model3_init]
name_l = ['sigmoid_model3', 'sigmoid_model3_init']

```

```python
train_l, test_l = model_comparison(model_l = model_l,
 name_l = name_l,
 train_data = train_loader,
 test_data = test_loader,
 num_epochs = 2,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)
```

相比于**sigmoid激活函数(本身生成正数**)，Xavier初始化方法更适用于tanh激活函数，核心原因在于tanh激活函数本身能够生成Zero-centered Data，配合Xavier初始化生成的参数，能够更好的确保各层梯度平稳、确保各层平稳学习。

![1691584000117]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_011.png' | relative_url }})

#### 7.2.2 PyTorch实现Xavier均匀分布的参数创建

类似的，我们可以使用torch.nn.init.xavier_normal_(tensor, gain=1.0)进行初始化参数设置。

$$\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$$

从理论上来说，均匀分布和高斯分布并没有根本性区别，二者任选其一使用即可。不过也有一些不是很严谨的实验证明了**均匀分布比高斯分布能够产生相对较大的梯度，因而模型学习效果会更好**。

### 7.3 Kaiming方法（HE初始化）

尽管Xavier初始化能够在Sigmoid和tanh激活函数叠加的神经网络中起到一定的效果，但由于**ReLU激活函数属于非饱和类激活函数**，并不会出现类似Sigmoid和tanh激活函数使用过程中可能存在的梯度消失或梯度爆炸问题，反而因为ReLU激活函数的不饱和特性，ReLU激活函数的叠加极有可能出现**神经元活性消失**的问题，很明显，**该类问题无法通过Xavier初始化解决**。

尽管如此，对参数的初始值进行合理设置，仍然是保证模型有效性的有效方法，同样也能一定程度上解决ReLU激活函数的神经元活性消失问题。目前通用的针对ReLU激活函数的初始化参数方法，是由何凯明在2015年的《Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification》一文中所提出的HE初始化方法，也被称为Kaiming方法。[原文地址](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)。

当然，He初始化也遵循Glorot条件，即参数初始化结果要求正向传播时各层接收到的数据方差保持一致、反向传播时各层参数梯度的方差保持一致，不过由于每一层的激活值（激活函数输出结果）均值不为0，因此Xavier的数学推导过程不再成立。关于HE初始化的数学推导此处不进行深入讲解，感兴趣的同学可自行参考论文中给出的推导过程。需要知道的是，经过一系列的推导和论证之后，HE初始化仍然是规定参数是满足均值为0的随机变量，并且仍然借助均匀分布和高斯分布进行随机变量创建，不同的是Xavier中参数方差为：
$$
Var(w)_{Xavier} = \frac{2}{fan_{in}+fan_{out}}
$$
而HE初始化过程中，参数方差为：
$$
Var(w)_{HE} = \frac{2}{fan_{in}} 或 Var(w)_{HE} = \frac{2}{fan_{out}}
$$
当然，根据参数方差，我们就能确定参数满足均匀分布时的边界，以及满足高斯分布时的基本形态。均匀分布时$bound = \sqrt{3var}$，因此参数分布区间为:$$(-\sqrt{\frac{6}{fan_{in}}}, \sqrt{\frac{6}{fan_{in}}})$$

后续还将介绍其他初始化方法，我们仍然可以从以下几个角度进行理解：

(1).为了满足或者更好的满足Glorot条件；

(2).数学理论推导出参数方差；

(3).方差由扇入和扇出神经元个数组成。

#### 7.3.1 PyTorch实现HE初始化的均匀分布参数创建

torch.nn.init.kaiming_uniform_进行初始化参数设置。

参数如下：

```python
nn.init.kaiming_uniform_(
 tensor,
 a=0,
 mode='fan_in',
 nonlinearity='leaky_relu',
)
```

- mode：参数表示选择带入扇入还是扇出的神经元个数进行计算，正如前文所说，理论上二者对建模没有明显影响，可任选其一，但实际由于模型个体差异，在实际使用过程中还是略有差异，我们可以根据实际效果进行选择；
- a：为使用ReLU变种激活函数时的修正系数；
- nonlinearity：表示所选用的变种ReLU激活函数类型，需要配合a参数使用，相关使用方法我们将在后续介绍ReLU变种激活函数的使用时一并介绍。

另外，值得注意的是，kaiming方法的帮助文档中的数学公式有误，切勿参考该公式进行建模。

帮助文档中公式：
$$
\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
$$
而实际上kaiming方法并没有gain增益系数，只有a的一个修正系数，实际公式如下：
$$
bound = \sqrt{\frac{6}{(1+a^2)fan_{in}}}
$$
注意：自己手动创建的tensor 张量列数表示$fan_{in}$的个数，行数表示$fan_{out}$的个数 。但是 PyTorch 每个线性层的张量行列排布与手动创建的是一个转置关系。

HE初始化如何帮助模型规避Dead ReLU Problem？核心在于模型初始化时如果参数完全随机选择，就有**可能出现初始参数全部输出0的结果，而通过HE初始化的参数不会出现上述情况**。

![1691586008822]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_012.png' | relative_url }})

我们发现，使用HE初始化之后模型收敛速度明显提升。

#### 7.3.2 PyTorch实现HE初始化的正态分布参数创建

`torch.nn.init.kaiming_normal_`来进行满足正态分布的初始化参数设置。

```python
nn.init.kaiming_normal_(
 tensor,
 a=0,
 mode='fan_in',
 nonlinearity='leaky_relu',
)
```

同样，说明文档中的公式存在错误。以下是说明文档中的公式:
$$
\text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
$$
实际上，PyTorch 中所使用的HE初始化时，初始参数的方差为：
$$
Var(w) = \frac{2}{(1+a^2)fan_{in}}
$$

### 7.4 参数初始化的作用局限

截止目前，我们讨论了围绕Glorot条件进行的参数初始化方法，当然，合理的设置初始参数值能够一定程度上使得模型各层都得到有效的学习，模型训练过程更加平稳、收敛速度也更快。但由于我们设置的是初始条件，伴随着模型不断训练，由于**受到激活函数导函数本身特性影响，仍然有可能在迭代过程中出现梯度不均衡的现象**。

因为Xavier初始化方法，由于是对于“初始值”的修改，因此也会存在**随着迭代次数增加**就逐渐破坏了Zero-Centered Data这一条件的问题

![1691586672651]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_013.png' | relative_url }})

权重衰退是最常见的一种处理过拟合的方法，是最广泛使用的正则化技术之一。权重衰退属于控制每个参数（取值/可选择的值）范围较小 的方法。

然而模型一旦开始训练，我们是不能手动参与修改模型参数的。那此时应该如何处理梯度不均衡的问题呢？我们知道，**影响梯度计算**的三个核心因素，分别是参数状态值w、激活值和输入的数据，参数状态值由模型迭代的数学过程决定，激活值很大程度上由我们所选取的激活函数决定，如果从Glorot条件入手，我们就只剩下一个可以人工修改的选项：**每一个线性层接收到的数据**(第一层输入为$X$，第二层输入为第一层激活函数后的数据$F(X*w_1)$……)。 因此做**输入数据的归一化**
$$
\begin{aligned}grad_1&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot w_2\cdot f(X*w_1)\cdot X\\\\grad_2&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot F(X*w_1)\\\\grad_3&=\frac{\partial loss}{\partial\hat{y}}\cdot F(F(X*w_1)*w_2)\end{aligned}
$$

---

### 7.5 优化算法
#### 1.优化问题

一般优化问题可写为：

$$
\min_x f(x)\quad \text{s.t.}\quad x\in C
$$

其中 $f:\mathbb{R}^n\to\mathbb{R}$ 为目标函数，$C$ 为约束集合。常见约束集合形式为：

$$
C=\{x\mid h_1(x)=0,\ldots,h_m(x)=0,\;g_1(x)\le 0,\ldots,g_k(x)\le 0\}
$$

若 $C=\mathbb{R}^n$，则是不受约束优化。

#### 2.局部最小 vs 全局最小

![image-20230902183043799]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_015.png' | relative_url }})

#### 3.凸集和凸函数

- 凸集：形象化来说，就是这个集合上任意两个点连一条线，这个线在集合里面
- 凸函数：形象上来说函数上任取两个点连线，函数都在该线下面
- 凸优化问题：局部最小一定是全局最小。严格凸优化问题有唯一的全局最小。
- 凸：线性回归，softmax回归
- 非凸：其他（MLP,CNN,RNN,attention）

#### 4.梯度下降

- 梯度下降——最简单的迭代求解算法
- 随机梯度下降
- 求导数需要求所有样本导数，样本多的情况下代价太大
- 理论依据：所用样本，和随机选取一个样本得到的数学期望是一样的。
- 小批量随机梯度下降（实际应用的）
- 计算原因：计算单样本的梯度难以完全利用硬件资源
- 采集一个随机子集
- 理论依据：无偏近，但降低了方差

#### 5.冲量法

- 使用平滑过的梯度对权重更新，不容易震荡
- momentum

![image-20230902184448051]({{ '/assets/learning/10_deep_learning_optimization_algorithms/figure_016.png' | relative_url }})

#### 6.Adam

- 非常平滑，对于学习率不敏感
- 对于t比较小的时候，由于$v_0=0$,所以会导致一开始值比较小，做了一个修正。

动量法的一阶动量为：

$$
v_t=\beta_1v_{t-1}+(1-\beta_1)g_t,\qquad \beta_1\approx0.9
$$

展开后可得：

$$
v_t=(1-\beta_1)\left(g_t+\beta_1g_{t-1}+\beta_1^2g_{t-2}+\cdots+\beta_1^{t-1}g_1\right)
$$

由于 $v_0=0$，训练初期会有偏差，因此使用偏差修正：

$$
\hat v_t=\frac{v_t}{1-\beta_1^t}
$$

- 对$v_t$修正是为了使梯度比较平滑

- 为什么除以$\sqrt{\widehat{s}_t}+\epsilon$？
- 在nlp里面常用，起到正则化的作用，控制每个维度的值在合适的大小。

Adam 中的二阶动量为：

$$
s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2,\qquad \beta_2\approx0.999
$$

偏差修正为：

$$
\hat s_t=\frac{s_t}{1-\beta_2^t}
$$

最终参数更新公式为：

$$
w_t=w_{t-1}-\eta\frac{\hat v_t}{\sqrt{\hat s_t}+\epsilon}
$$

### 总结

- 深度学习模型大部分是非凸的
- 小批量随机梯度下降是最常见的优化算法
- 冲量是对梯度做平滑
- Adam是对梯度做平滑，且对梯度各个维度值做重新调整，对于学习率不敏感，也就是说学习率的可调参范围更大，小批量随机梯度下降对应的可调参返回过更小。
