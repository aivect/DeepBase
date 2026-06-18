---
layout: article
title: "机器学习目标与模型评估"
key: learning-08-model-evaluation
permalink: /learning/08-model-evaluation.html
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

## 1 机器学习目标与模型评估方法

首先，在统计分析领域，我们会假设现在的数据和未来的数据其实都属于某个存在但不可获得的总体，也就是说，现在和未来的数据都是从某个总体中抽样而来的，都是这个总体的样本。而正式因为这些数据属于同一个总体，因此具备某些相同的规律，而现在挖掘到的数据规律也就在某些程度上可以应用到未来的数据当中去，不过呢，不同抽样的样本之间也会有个体之间的区别，另外模**型本身也无法完全捕获规律，而这些就是误差的来源**。

虽然样本和总体的概念是统计学概念，但样本和总体的概念所假设的前后数据的“局部规律一致性”，却是所有机器学习建模的基础。试想一下，如果获取到的数据前后描绘的不是一件事情，那么模型训练也就毫无价值（比如拿着A 股走势预测的时间序列预测某地区下个季度患病人次）。因此，无论是机器学习所强调的从业务角度出发，要确保前后数据描述的一致性，还是统计分析所强调的样本和总体的概念，都是建模的基础。

![1691475770962]({{ '/assets/learning/08_model_evaluation/figure_001.png' | relative_url }})

而对于机器学习来说，并没有借助“样本-总体”的基本理论，而是简单的采用了一种后验的方法来判别模型有效性，前面说到，我们**假设前后获取的数据拥有规律一致性，但数据彼此之间又略有不同**，为了能够**在捕捉规律的同时又能考虑到“略有不同”所带来的误差**，机器学习会把当前能获取到的数据划分成**训练集(trainSet)和测试集(testSet)**

**在训练集上构建模型，然后带入测试集的数据，观测在测试集上模型预测结果和真实结果之间的差异**。这个过程其实就是在模拟获取到真实数据之后模型预测的情况，此前说到，模型能够在未知标签的数据集上进行预测，就是模型的核心价值，此时的测试集就是用于模拟未来的未知标签的数据集。如果模型能够在测试集上有不错的预测效果，我们就“简单粗暴”的认为模型可以在真实的未来获取的未知数据集上有不错的表现。其一般过程可以由下图表示。

![1691475943891]({{ '/assets/learning/08_model_evaluation/figure_002.png' | relative_url }})

相比数理统计分析，确实没有“那么严谨”，但更易于理解的理论和更通用的方法，却使得机器学习可以在更为广泛的应用场景中发挥作用。（当然，负面影响却是，机器学习在曾经的很长一段时间内并不是主流的算法。）

据此，我们称模型在训练集上误差称为训练误差，在测试集上的误差称为泛化误差。

不过毕竟在测试集上进行测试还只是模拟演习，我们采用模型的泛化能力来描述模型在未知数据上的判别能力，当然泛化能力无法准确衡量（未知的数据还未到来，到来的数据都变成了已知数据），**我们只能通过模型在训练集和测试集上的表现，判别模型泛化能力**，当然，就像此前说的一样，最基本的，我们**会通过模型在测试集上的表现来判断模型的泛化能力**。

## 2 手动实现训练集和测试集切分

据此，我们称模型在训练集上误差称为训练误差，在测试集上的误差称为泛化误差，不过毕竟在测试集上进行测试还只是模拟演习，我们采用模型的泛化能力来描述模型在未知数据上的判别能力，当然泛化能力无法准确衡量（未知的数据还未到来，到来的数据都变成了已知数据），我们只能通过模型在训练集和测试集上的表现，判别模型泛化能力，当然，就像此前说的一样，最基本的，我们会通过模型在测试集上的表现来判断模型的泛化能力。

创建一个**切分训练集和测试集的函数** ：

```python
def data_split(features, labels, rate=0.7):
    """
 训练集和测试集切分函数

 :param features: 输入的特征张量
 :param labels：输入的标签张量
 :param rate：训练集占所有数据的比例
 :return Xtrain, Xtest, ytrain, ytest：返回特征张量的训练集、测试集，以及标签张量的训练集、测试集
 """
    num_examples = len(features) # 总数据量
    indices = list(range(num_examples)) # 数据集行索引
    random.shuffle(indices) # 乱序调整
    num_train = int(num_examples * rate) # 训练集数量
    indices_train = torch.tensor(indices[: num_train]) # 在已经乱序的indices中挑出前num_train数量的行索引值
    indices_test = torch.tensor(indices[num_train: ])
    Xtrain = features[indices_train] # 训练集特征
    ytrain = labels[indices_train] # 训练集标签
    Xtest = features[indices_test] # 测试集特征
    ytest = labels[indices_test] # 测试集标签
    return Xtrain, Xtest, ytrain, ytest
```

因此，根据经验，我们一般来说会按照8：2或7：3比例进行划分。

值得一提的是，在机器学习领域，充斥着大量的“经验之谈”或者“约定俗成”的规则，一方面这些经验为建模提供了诸多便捷、也节省了很多算力，但另一方面，通过经验来决定影响模型效果的一些“超参数”取值的不严谨的做法，也被数理统计分析流派所诟病。

```python
torch.manual_seed(420)

# 初始化数据
features, labels = tensorGenReg()

# 切分训练集和测试集
Xtrain, Xtest, ytrain, ytest = data_split(features, labels)

# 初始化核心参数
batch_size = 10 # 小批的数量
lr = 0.03 # 学习率
num_epochs = 5 # 训练过程遍历几次数据
w = torch.zeros(3, 1, requires_grad = True) # 随机设置初始权重

# 参与训练的模型方程
def linreg(X, w):
    return torch.mm(X, w)

def MSE_loss(y_hat, y):
    num_ = y.numel()
    sse = torch.sum((y_hat.reshape(-1, 1) - y.reshape(-1, 1)) ** 2)
    return sse / num_ #平均损失值
def sgd(params, lr):
    params.data -= lr * params.grad
    params.grad.zero_()

    net = linreg # 使用回归方程
    loss = MSE_loss # 均方误差的一半作为损失函数

    # 模型训练过程
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, Xtrain, ytrain):
            l = loss(net(X, w), y)
            l.backward()
            sgd(w, lr)
```

查看模型在训练集、测试集上的MSE

```python
MSE_loss(torch.mm(Xtrain, w), ytrain)
'''tensor(0.0001, grad_fn=<DivBackward0>)'''
```

```python
MSE_loss(torch.mm(Xtest, w), ytest)
'''tensor(9.8789e-05, grad_fn=<DivBackward0>)'''
```

## 3 Dataset和DataLoader基本使用方法与数据集切分函数

### 3.1 Dataset 和 DataLoader 的基本使用方法

random_split(带切分张量，[num_train, num_test])随机切分函数

首先，在 PyTorch 的`torch.utils.data`中，提供了`random_split`函数可用于数据集切分。

```python
random_split(t, [2, 2])
```

`random_split`函数其实生成了生成器切分结果的生成器，并不是和此前定义的函数一样，直接切分数据后返回。当然这也符合utils.data模块主要生成映射式和迭代式对象的一般规定。

tensorDataSet是对张量类型特征、标签数据进行封装

由于在大多数调库建模过程中，我们都是先通过创建Dataset的子类并将数据保存为该子类类型，然后再使用DataLoader进行数据载入，因此更为通用的做法是先利用Dataset和DatasetLoader这两个类进行数据的读取、预处理和载入，然后再使用random_split函数进行切分。

- 创建数据类

**PyTorch 中所有的数据都是Dataset的子类**，**换而言之就是在使用 PyTorch 建模训练数据时，需要创建一个和数据集对应的类来表示该数据集**，此前我们使用的 TensorDataset函数其实就是一个简单的类型转化函数，将数据统一转化为“TensorDataset”类然后带入模型进行计算。

**而 TensorDataset其实使用面较窄，最直接的限制就是该函数只能将张量类型转化为 TensorDataset类**

而 TensorDataset其实使用面较窄，最直接的限制就是该函数只能将张量类型转化为 TensorDataset类

再次强调，Dataset类主要负责数据类的生成，在 PyTorch 中，所有数据集都是Dataset的子类；而DatasetLoader类则是加载模型训练的接口，二者基本使用流程如下：

![1691478200575]({{ '/assets/learning/08_model_evaluation/figure_003.png' | relative_url }})

接下来，创建一个用于表示该数据集的Dataset的子类。在创建Dataset的子类过程中，必须要重写__getitem__方法和__len__方法，其中__getitem__方法返回输入索引后对应的特征和标签，而__len__方法则返回数据集的总数据个数。当然，在必须要进行的__init__初始化过程中，我们也可输入可代表数据集基本属性的相关内容，包括数据集的特征、标签、大小等等，视情况而定。

```python
class LBCDataset(Dataset):
    def __init__(self,data): # 创建该类时需要sklearn输入导入的数据集
        self.features = data.data # features属性返回数据集特征
        self.labels = data.target # labels属性返回数据集标签
        self.lens = len(data.data) # lens属性返回数据集大小
        def __getitem__(self, index):
            # 调用该方法时需要输入index数值，方法最终返回index对应的特征和标签
            return self.features[index,:],self.labels[index]
        def __len__(self):
            # 调用该方法不需要输入额外参数，方法最终返回数据集大小
            return self.lens
```

封装好的数据可以直接进行索引，并且能够返回实体结果

```python
from sklearn.datasets import load_breast_cancer as LBC
data = LBC()
LBC_data = LBCDataset(data)
LBC_data[:]
```

- 使用random_split方法对其进行切分 :

```python
LBC_train, LBC_test = random_split(LBC_data, [num_train, num_test])
```

注，此时切分的结果是一个映射式的对象，只有dataset和indices两个属性，

其中dataset属性用于切分前的查看原数据集对象，indices属性用于查看切分后数据集的每一条数据的index（序号），这里的index是对应原数据集的索引。

### 3.2 dataLoader

然后使用DataLoader函数进行数据转化，由一般数据状态转化为“可建模”的状态。所谓“可建模”状态，指的是经过DataLoader处理的数据，不仅包含数据原始的数据信息，还包含数据处理方法信息，如调用几个线程进行训练、分多少批次等，DataLoader常用参数如下：

- batch_size:每次迭代输入多少数据，如果是小批量梯度下降，则输入的数据量就是小批量迭代过程中“小批”的数量
- shuffle:是否需要先打乱顺序然后再进行小批量的切分，一般训练集需要乱序，而测试集乱序没有意义
- num_worker:启动多少线程进行计算

**对于测试集来说，数据装载并不是一定要进行的，如果测试集只是用于检测模型效果，有时可以不用装载直接带入计算。**

同样，经过DataLoader处理后的数据也可以使用dataset属性查看原数据。

### 2.3 建模及评估过程

- 数据准备过程

```python

from torch.utils.data import random_split
# 生成数据
features, labels = tensorGenReg()
features = features[:, :-1] # 剔除最后全是1的列

# 创建一个针对手动创建数据的数据类
class GenData(Dataset):
    def __init__(self, features, labels): # 创建该类时需要输入的数据集
        self.features = features # features属性返回数据集特征
        self.labels = labels # labels属性返回数据集标签
        self.lens = len(features) # lens属性返回数据集大小

        def __getitem__(self, index):
            # 调用该方法时需要输入index数值，方法最终返回index对应的特征和标签
            return self.features[index,:],self.labels[index]

        def __len__(self):
            # 调用该方法不需要输入额外参数，方法最终返回数据集大小
            return self.lens

        # 实例化对象
        data = GenData(features, labels)

        # 切分数据集
        num_train = int(data.lens * 0.7)
        num_test = data.lens - num_train
        data_train, data_test = random_split(data, [num_train, num_test])

        # 加载数据
        train_loader = DataLoader(data_train, batch_size=10, shuffle=True)
        test_loader = DataLoader(data_test, batch_size=10, shuffle=False)
```

- 构建模型

```python
# 初始化核心参数
batch_size = 10 # 小批的数量
lr = 0.03 # 学习率
num_epochs = 3 # 训练过程遍历几次数据

# Stage 1.定义模型
class LR(nn.Module):
    def __init__(self, in_features=2, out_features=1): # 定义模型的点线结构
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        def forward(self, x): # 定义模型的正向传播规则
            out = self.linear(x)
            return out

        # 实例化模型
        LR_model = LR()

        # Stage 2.定义损失函数
        criterion = nn.MSELoss()

        # Stage 3.定义优化方法
        optimizer = optim.SGD(LR_model.parameters(), lr = 0.03)

        # Stage 4.模型训练与测试
        def fit(net, criterion, optimizer, batchdata, epochs=3):
            for epoch in range(epochs):
                for X, y in batchdata:
                    yhat = net.forward(X)
                    loss = criterion(yhat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
```

- 模型训练与测试

```python
fit(net = LR_model,
 criterion = criterion,
 optimizer = optimizer,
 batchdata = train_loader,
 epochs = num_epochs
)
```

- 在训练集的损失：

```python
# 计算训练集MSE
F.mse_loss(LR_model(data[data_train.indices][0]), data[data_train.indices][1])
```

- 在训练集的损失：

```python
# 计算测试集MSE
F.mse_loss(LR_model(data[data_test.indices][0]), data[data_test.indices][1])
```

## 4 实用函数补充

- 数据封装、切分和加载函数

该函数可以直接处理此前定义的数据生成器创建的数据。

```python
def split_loader(features, labels, batch_size=10, rate=0.7):
    """数据封装、切分和加载函数：

 :param features：输入的特征
 :param labels: 数据集标签张量
 :param batch_size：数据加载时的每一个小批数据量
 :param rate: 训练集数据占比
 :return：加载好的训练集和测试集
 """
    data = GenData(features, labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return(train_loader, test_loader)
```

```python
# 创建一个针对手动创建数据的数据类
class GenData(Dataset):
    def __init__(self, features, labels): # 创建该类时需要输入的数据集
        self.features = features # features属性返回数据集特征
        self.labels = labels # labels属性返回数据集标签
        self.lens = len(features) # lens属性返回数据集大小

        def __getitem__(self, index):
            # 调用该方法时需要输入index数值，方法最终返回index对应的特征和标签
            return self.features[index,:],self.labels[index]

        def __len__(self):
            # 调用该方法不需要输入额外参数，方法最终返回数据集大小
            return self.lens

```

- 模型训练函数

模型训练函数并不是新的函数，此处正式对其进行定义并写入自定义模块中，方便后续调用。

cla=False：是否是分类问题

```python
def fit(net, criterion, optimizer, batchdata, epochs=3, cla=False):
    """模型训练函数

 :param net：待训练的模型
 :param criterion: 损失函数
 :param optimizer：优化算法
 :param batchdata: 训练数据集
 :param cla: 是否是分类问题
 :param epochs: 遍历数据次数
 """
    for epoch in range(epochs):
        for X, y in batchdata:
            if cla True:
                y = y.flatten().long() # 如果是分类问题，需要对y进行整数转化
                yhat = net.forward(X)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

- MSE计算函数

接下来，我们借助F.mse_loss，定义一个可以直接根据模型输出结果和加载后的数据计算MSE的函数。

还原Dataset类：，data[:]是获取第一维度的所有样本中，data[:] [0]是获取第一维度的所有样本中的特征张量

```python
X = data[:][0]
```

```python
def mse_cal(data_loader, net):
    """mse计算函数

 :param data_loader：dataloader加载好的数据
 :param net: 模型
 :return：根据输入的数据，输出其MSE计算结果
 """
    data = data_loader.dataset # 还原Dataset类
    X = data[:][0] # 还原数据的特征
    y = data[:][1] # 还原数据的标签
    yhat = net(X)
    return F.mse_loss(yhat, y)
```

```python
mse_cal(train_loader, LR_model) # 计算训练误差
```

- 准确率计算函数

```python
def accuracy_cal(data_loader, net):
    """准确率

 :param data_loader：加载好的数据
 :param net: 模型
 :return：根据输入的数据，输出其准确率计算结果
 """
    data = data_loader.dataset # 还原Dataset类
    X = data[:][0] # 还原数据的特征
    y = data[:][1] # 还原数据的标签
    zhat = net(X) # 默认是分类问题，并且输出结果是未经softmax转化的结果zhat
    soft_z = F.softmax(zhat, 1) # 进行softmax转化
    acc_bool = torch.argmax(soft_z, 1).flatten() y.flatten()
    acc = torch.mean(acc_bool.float())
    return acc
```
注：**vargmax(soft_z, 1)返回soft_z每一行的最大值的索引，每个索引代表一个标签。**
