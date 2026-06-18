---
layout: article
title: "深度学习基础网络搭建与实现"
key: learning-07-pytorch-network-implementation
permalink: /learning/07-pytorch-network-implementation.html
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

类似 PyTorch 的这种，看似对初学者略显复杂的建模流程，实际上都是为了能够更好的满足深度学习建模的一般情况：针对非结构化数据、在超大规模的数据集上进行模型训练。我们前面提到 PyTorch 在读取数据的过程中需要使用Dataset和DataLoader数据进行封装和加载，其实是为了能够实现数据的迭代式存储和映射式存储，也就是通过生成数据的生成器或者保存数据的映射关系，来避免数据的重复存储，如进行小批量数据划分时，PyTorch 并未真正意义上的把数据进行切分然后单独存储，而是创建了每个“小批”数据和原数据的映射关系（或者说“小批”数据的索引值），然后借助这种映射关系，在实际需要使用这些数据的时候在对其进行提取。这么做的原因，当然也是因为当进行海量数据处理时，划分多个数据集进行额外的存储显然是不合适的。当然 PyTorch 在设计的时候将很多数据预处理的方法也放到了数据封装和加载的流程中，这些我们将在后面谈到。而在 PyTorch 的建模过程中，类的频繁使用其实也是为了能够更加灵活的创建不同类型的神经网络模型。

练掌握深度学习建模流程、熟练使用基本函数和类却是后续学习的基础

## 1 深度学习建模实验中数据集生成函数的创建与使用

- 导入相关的包

```python
# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# PyTorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
```

### 1.1 回归类数据集创建方法

#### 1.1.1 手动生成数据

回归类模型的数据，特征和标签都是连续型数值。

正常情况，应该是对于连续型数值标签的预测，我们采用回归类模型，此处因为先生成数据后进行建模，因此我们称可用于回归模型训练的数据为回归类模型数据，分类模型数据亦然

数据生成：生成两个特征、存在偏差，自变量和因变量存在线性关系的数据集

#### 1.1.2 创建生成回归类数据的函数

```python
def tensorGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

 :param num_examples: 创建数据集的数据量
 :param w: 包括截距的（如果存在）特征系数向量
 :param bias：是否需要截距
 :param delta：扰动项取值
 :param deg：方程次数
 :return: 生成的特征张和标签张量
 """

    if bias True:
        num_inputs = len(w)-1 # 特征张量
        features_true = torch.randn(num_examples, num_inputs) # 不包含全是1的列的特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float() # 自变量系数
        b_true = torch.tensor(w[-1]).float() # 截距
        if num_inputs 1: # 若输入特征只有1个，则不能使用矩阵乘法
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
            features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1) # 在特征张量的最后添加一列全是1的列
            labels = labels_true + torch.randn(size = labels_true.shape) * delta

        else:
            num_inputs = len(w)
            features = torch.randn(num_examples, num_inputs)
            w_true = torch.tensor(w).reshape(-1, 1).float()
            if num_inputs 1:
                labels_true = torch.pow(features, deg) * w_true
            else:
                labels_true = torch.mm(torch.pow(features, deg), w_true)
                labels = labels_true + torch.randn(size = labels_true.shape) * delta
                return features, labels
```

### 1.2 分类数据集创建方法

#### 1.2.1 创建生成分类数据的函数

```python
def tensorGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """分类数据集创建函数。

 :param num_examples: 每个类别的数据数量
 :param num_inputs: 数据集特征数量
 :param num_class：数据集标签类别总数
 :param deg_dispersion：数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
 :param bias：建立模型逻辑回归模型时是否带入截距
 :return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组。
 """

    cluster_l = torch.empty(num_examples, 1) # 每一类标签张量的形状
    mean_ = deg_dispersion[0] # 每一类特征张量的均值的参考值
    std_ = deg_dispersion[1] # 每一类特征张量的方差
    lf = [] # 用于存储每一类特征张量的列表容器
    ll = [] # 用于存储每一类标签张量的列表容器
    k = mean_ * (num_class-1) / 2 # 每一类特征张量均值的惩罚因子（视频中部分是+1，实际应该是-1）

    for i in range(num_class):
        data_temp = torch.normal(i*mean_-k, std_, size=(num_examples, num_inputs)) # 生成每一类张量
        lf.append(data_temp) # 将每一类张量添加到lf中
        labels_temp = torch.full_like(cluster_l, i) # 生成类一类的标签
        ll.append(labels_temp) # 将每一类标签添加到ll中

        features = torch.cat(lf).float()
        labels = torch.cat(ll).long()

        if bias True:
            features = torch.cat((features, torch.ones(len(features), 1)), 1) # 在特征张量中添加一列全是1的列
            return features, labels
```

#### 1.2.2 创建小批量切分函数

批量切分函数的目标就是根据设置的“批数”，将原数据集随机均匀切分。可通过如下函数实现：

```python
def data_iter(batch_size, features, labels):
    """
    数据切分函数

    :param batch_size: 每个子数据集包含多少数据
    :param featurs: 输入的特征张量
    :param labels：输入的标签张量
    :return l：包含batch_size个列表，每个列表切分后的特征和标签所组成
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        l.append([torch.index_select(features, 0, j), torch.index_select(labels, 0, j)])
        return l
```

## 2 安装TensorboardX

### 2.1 SummaryWriter类与Tensorboard的基本使用

1）实例化summarywriter

```python
writer = SummaryWriter(log_dir='test')
```

实例化的过程中需要确定保存路径，当保存在主目录文件内时，只需输入文件夹名称即可。当文件夹不存在时，系统会自动创建一个。每次实例化一次SummaryWriter，就会在对应文件夹内新生成一个文件。单个文件夹内允许存在多个文件，在实际读取过程中会一次性读取文件夹内全部文件。

```python
writer.log_dir # 通过log_dir查看summary对象记录文件的位置
```

2）记录数据

```python
for i in range(10):
 writer.add_scalar('mul', i*i, i)
```

此处我们使用了add_scalar方法进行数据记录，该方法用于记录一组组标量，在实际记录过程中，通过迭代的方式逐个增加。其中，第一个参数代表改组记录的名称（也就是生成图像的名称），第二个参数代表y值，第三个参数代表x值。完成记录后，即可读取文件查看记录结果。除此以外，常用的还有add_graph方法用于记录深度学习建模流程。

3）启动服务读取文件

cd "F:\Code File\PyTorch 实战\"

tensorboard --logdir="test"

接下来，在本地启动服务，读取刚刚生成的记录文件。同样，我们可以在jupyter中的命令行直接启动服务。

注意，如果不是通过jupyter，而是通过cmd打开的命令行，如果要进入其他盘的主目录，例如F盘，则需要先输入F:指令，将当前操作主目录转移到F盘上。

4）进入6006端口页面

如果默认6006端口冲突，也可自行修改服务端口号，只需在启动命令后后输入参数--port XXXX即可

## 3 线性回归建模实验

### 3.1 深度学习建模流程

#### Stage 1 模型选择

| 项目     | 内容                                                       |
| -------- | ---------------------------------------------------------- |
| 任务描述 | 在深度学习领域，模型选择过程就是确定神经网络的基本结构     |
| 实际工作 | 确定神经网络的层数、每一层神经元的个数，以及选择激活函数。 |

#### Stage 2 确定目标函数
| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 任务描述 | 在确定模型基本结构之后，根据模型实际情况确定目标函数         |
| 实际工作 | 构建一个包含模型参数的函数方程、并且方程取值和建模目标一致，大多数情况下，我们都是沿着方程极小值求解 |

#### Stage 3 选择优化方法
| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 任务描述 | 围绕目标函数进行最小值求解                                   |
| 实际工作 | 根据损失函数的函数特性，兼顾实际算力的消耗，选择一项最优化工具，往往也就是某项数学工具 |

#### Stage 4 模型训练
| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 任务描述 | 将模型训练至可用                                             |
| 实际工作 | 利用优化方法，求解损失函数，并得到一组模型参数。对应在神经网络中，就是得到一组连接神经元的边的参数取值 |



反向传播过程：

![1691452641244]({{ '/assets/learning/07_pytorch_network_implementation/figure_002.png' | relative_url }})

那backward函数是怎么判断具体求解哪个对象的梯度的呢？

- 其实就是靠requires_grad。首先**backward值会识别叶子节点，不在叶子上的变量是不会被backward考 虑的**。对于全部叶子节点来说，只有属性requires_grad=True的节点，才会被计算。在设置X与y 时， 我们都没有写requires_grad参数，也就是默认让“允许求解梯度”这个选项为False，所以backward在计 算的时候就只会计算关于w的部分。

关于反向传播上可微张量的in-place operation（对原对像修改操作）的相关讨论 。

(1)正常情况下，可微张量的in-place operation会导致系统无法区分叶节点和其他节点的问题

PyTorch 中不允许叶节点使用in-place operation，根本原因是会造成叶节点和其他节点类型混乱。不过，虽然可微张量不允许in-place operation，但却可以通过其他方法进行对w进行修改。

```python
w = torch.tensor(2., requires_grad = True)
w = w * 2
```

不过此时，w就不再是叶节点了， w已经成为输出节点。我们也无法通过反向传播求其导数 。

而在一张计算图中，缺少了对叶节点反向传播求导数的相关运算，计算图也就失去了核心价值。因此在实际操作过程中，应该尽量避免导致叶节点丢失的相关操作。

(2) 叶节点数值修改方法：

当然，如果出现了一定要修改叶节点的取值的情况，典型的如梯度下降过程中利用梯度值修改参数值时，可以使用此前介绍的暂停追踪的方法，如使用with torch.no_grad()语句或者torch.detach()方法，使得修改叶节点数值时暂停追踪，然后再生成新的叶节点带入计算，如：

**detach_()是生成对象并替代w**

```python
w = torch.tensor(2., requires_grad = True)
w

# 利用with torch.no_grad()暂停追踪
with torch.no_grad():
 w -= w * 2
w

# 利用detach生成新变量。‘detach_()’是生成对象并替代w
w.detach_()
w -= w * 2
w.requires_grad = True
```

(3) 当然，此处我们介绍另一种方法，.data来返回可微张量的取值，从在避免在修改的过程中被追踪

```python
w = torch.tensor(2., requires_grad = True)
w.data -= w * 2 # 对其数值进行修改，不受可微影响
```

- Stage 1.模型选择

```python
def linreg(X, w):
 return torch.mm(X, w)
```

- Stage 2.确定目标函数

和此前一样，我们使用MSE作为损失函数，也就是目标函数

```python
def squared_loss(y_hat, y):
 num_ = y.numel()
 sse = torch.sum((y_hat.reshape(-1, 1) - y.reshape(-1, 1)) ** 2)
 return sse / num_ #平均损失值
```

- Stage 3.定义优化算法

此处我们采用小批量梯度下降进行求解，每一次迭代过程都是(参数-学习率*梯度)。

params.grad.zero_()：梯度清零

params.data -= lr * params.grad

```python
def sgd(params, lr):
 params.data -= lr * params.grad
 params.grad.zero_()
```

小批量切分函数：

```python
def data_iter(batch_size, features, labels):
    """
 数据切分函数

 :param batch_size: 每个子数据集包含多少数据
 :param featurs: 输入的特征张量
 :param labels：输入的标签张量
 :return l：包含batch_size个列表，每个列表切分后的特征和标签所组成
 """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        l.append([torch.index_select(features, 0, j), torch.index_select(labels, 0, j)])
        return l
```

- Stage.4 训练模型

```python
# 设置随机数种子
torch.manual_seed(420)

# 初始化核心参数
batch_size = 10 # 每一个小批的数量
lr = 0.03 # 学习率
num_epochs = 3 # 训练过程遍历几次数据
w = torch.zeros(3, 1, requires_grad = True) # 随机设置初始权重

# 参与训练的模型方程
net = linreg # 使用回归方程
loss = squared_loss # MSE作为损失函数

# 模型训练过程
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w), y)
        l.backward()
        sgd(w, lr)
        # 这里train_l中的features features都为真实数值，则可以求得loss为数值
        train_l = loss(net(features, w),features)
        print('epoch %d, loss %f' % (epoch + 1, train_l))
```

### 3.2 快速实现

- 定义核心参数

```python
batch_size = 10 # 每一个小批的数量
lr = 0.03 # 学习率
num_epochs = 3 # 训练过程遍历几次数据

```

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建数据集
features, labels = tensorGenReg()
features = features[:, :-1] # 剔除最后全是1的列
data = TensorDataset(features, labels) # 数据封装
batchData = DataLoader(data, batch_size = batch_size, shuffle = True) # 数据加载
```

- Stage 1.定义模型

```python
class LR(nn.Module):
    def __init__(self, in_features=2, out_features=1): # 定义模型的点线结构
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        def forward(self, x): # 定义模型的正向传播规则
        out = self.linear(x)
        return out

# 实例化模型
LR_model = LR()
```

- Stage 2.定义损失函数

```python
criterion = nn.MSELoss()
```

- Stage 3.定义优化方法

```python
optimizer = optim.SGD(LR_model.parameters(), lr = 0.03)
```

- Stage 4.模型训练

```python
def fit(net, criterion, optimizer, batchdata, epochs):
    for epoch in range(epochs):
        for X, y in batchdata:
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, global_step=epoch)
```

- 开始训练

```python
# 设置随机数种子
torch.manual_seed(420)

fit(net = LR_model,
 criterion = criterion,
 optimizer = optimizer,
 batchdata = batchData,
 epochs = num_epochs)
```

计算MSE损失

```python
# 计算MSE
criterion(LR_model(features), labels)
```

当然，我们也可以通过add_graph方法，在writer中添加上述模型的记录图：

参数1是模型，参数2是元组，中间是训练数据

```python
writer.add_graph(LR_model, (features,))
```

## 4 逻辑回归建模实验

```python
# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# PyTorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

# 自定义模块
from torchLearning import *

# 导入以下包从而使得可以在jupyter中的一个cell输出多个结果
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

### 4.1 逻辑回归手动实现

#### 4.1.1 生成数据集

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建数据集
features, labels = tensorGenCla(num_class=2, bias=True)

# 可视化展示
plt.scatter(features[:, 0], features[:, 1], c = labels)
```

#### 4.1.2 建模过程

- Stage 1.模型选择

针对二分类问题（0-1问题），我们可以简单的输出一个结果(单层神经网络)

1）激活函数

```python
def sigmoid(z):
 return 1/(1+torch.exp(-z))
```

2）逻辑回归模型

```python
def logistic(X, w):
 return sigmoid(torch.mm(X, w))
```

3）辅助函数

由于sigmoid输出结果是连续值，而用于二分类判别时，我们需要将连续数值转化为所判定的类别，可定义对应分类函数如下：

```python
def cal(sigma, p=0.5):
 return((sigma >= p).float())
```

另外，对分类模型，我们往往会通过准确率判别模型效果，因此还需要定义准确率函数

```python
def accuracy(sigma, y):
    acc_bool = cal(sigma).flatten() y.flatten()
    acc = torch.mean(acc_bool.float())
    return(acc)
```

```python
p = torch.tensor([1, 1, 2]) torch.tensor([1, 2, 2])
'''tensor([ True, False, True])'''
p.float()
torch.mean(p.float())
'''tensor([1., 0., 1.])
tensor(0.6667)'''
```

- Stage 2.定义损失函数

```python
def cross_entropy(sigma, y):
    return(-(1/y.numel())*torch.sum((1-y)*torch.log(1-sigma)+y*torch.log(sigma)))
```

- Stage 3.定义优化方法

```python
def sgd(params, lr):
    params.data -= lr * params.grad
    params.grad.zero_()
```

- Stage 4.训练模型

```python
# 设置随机数种子
torch.manual_seed(420)

# 初始化核心参数
batch_size = 10 # 每一个小批的数量
lr = 0.03 # 学习率
num_epochs = 3 # 训练过程遍历几次数据
w = torch.ones(3, 1, requires_grad = True) # 随机设置初始权重

# 参与训练的模型方程
net = logistic # 使用逻辑回归方程
loss = cross_entropy # 交叉熵损失函数

# 训练过程
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w), y)
        l.backward()
        sgd(w, lr)
        train_acc = accuracy(net(features, w), labels)
        print('epoch %d, accuracy %f' % (epoch + 1, train_acc))
```

### 4.2 逻辑回归的快速实现

#### 4.2.1 构建模型

接下来，我们练习使用 PyTorch 中的函数和类，进行逻辑回归的快速构建。

- 定义核心参数

```python
batch_size = 10 # 每一个小批的数量
lr = 0.03 # 学习率
num_epochs = 3 # 训练过程遍历几次数据
```

- 数据准备

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建数据集
features, labels = tensorGenCla(num_class=2)
labels = labels.float() # 损失函数要求标签也必须是浮点型
data = TensorDataset(features, labels)
batchData = DataLoader(data, batch_size = batch_size, shuffle = True)
```

- Stage 1.定义模型

```python
class logisticR(nn.Module):
    def __init__(self, in_features=2, out_features=1): # 定义模型的点线结构
        super(logisticR, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        def forward(self, x): # 定义模型的正向传播规则
        out = self.linear(x)
        return out

# 实例化模型和
torch.manual_seed(420)
logic_model = logisticR()
```

- Stage 2.定义损失函数

```python
criterion = nn.BCEWithLogitsLoss()
```

- Stage 3.定义优化方法

```python
optimizer = optim.SGD(logic_model.parameters(), lr = lr)
```

- Stage 4.模型训练

```python
def fit(net, criterion, optimizer, batchdata, epochs):
    for epoch in range(epochs):
        for X, y in batchdata:
            zhat = net.forward(X)
            loss = criterion(zhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

和线性回归相同，由于上述模型只有一层，因此也可以通过nn.Linear(2, 1)函数直接建模。由于我们所采用的`BCEWithLogitsLoss`类进行的损失函数求解，该类会自动对输入对象进行sigmoid转化，因此上述过程和线性回归过程没有区别。

模型训练

```python
# 设置随机数种子
torch.manual_seed(420)

fit(net = logic_model,
 criterion = criterion,
 optimizer = optimizer,
 batchdata = batchData,
 epochs = num_epochs)
```

计算交叉熵损失

以下：logic_model(features)会根据输入的特征，去更具训练过的最后w，去重新正向传播算出zhat。

```python
# 计算交叉熵损失
criterion(logic_model(features), labels)
```

若需要完全复现模型训练过程，则需要在实例化之前设置随机数种子，或者在上一个随机数种子之前规定有限次的随机次数

- torch.manual_seed不会对random中的随机过程造成影响

- python的random中可以通过设置random.seed来控制随机过程

## 5 softmax回归建模实验

### 5.1 softmax回归手动实现

我们都知道，softmax是用于挑选最大值的一种方法，通过以下公式对不同类的计算结果进行数值上的转化
$$
\delta_k=\frac{e^{zk}}{\Sigma^Ke^k}
$$
softmax可以将结果放缩到0-1之间，并且使用softmax进行最大值的比较，相比max（softmax是max的柔化版本），能有效避免损失函数求解时在0点不可导的问题。

![1691474233920]({{ '/assets/learning/07_pytorch_network_implementation/figure_003.png' | relative_url }})
