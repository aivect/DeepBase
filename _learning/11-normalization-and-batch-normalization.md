---
layout: article
title: "数据归一化与 Batch Normalization"
key: learning-11-normalization-and-batch-normalization
permalink: /learning/11-normalization-and-batch-normalization.html
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

## 1 数据归一化与Batch Normalization理论基础

数据归一化是机器学习领域的一种对数据处理的常规方式。在传统机器学习领域，由于各特征的量**纲不一致**，可能出现建模过程中量纲较大的特征得不到有效学习的情况，而通过归一化处理之后的数据可以统一放缩在一个区间内，从而避免的各量纲的学习偏差问题，并且我们发现，**归一化处理之后的数据能够提升模型训练效率、加快模型收敛速度、提升模型稳定性**。当然，在传统机器学习领域，有很多需要确保模型可解释的情况，而**对数据进行归一化处理会降低模型本身的可解释性。**

量纲较大的特征得不到有效学习的情况：对于两个特征，一个x1是单位m，一个x2是km，对$y = w_1x_1+w_2x_2$，量纲较大则的x2取值会小，量纲较小则的x1则取值会大，而线性回归建模损失函数一般为MSE、SSE，显然这个会受到量纲较小的取值最大，

而在深度学习领域，正如上一节所论述的一样，将数据处理成Zero-centered Data，将能够有效确保模型各层学习的有效性，缓解梯度消失和梯度爆炸的情况发生，并且**深度学习并不要求可解释性**，因此数据标准化并不存在太多障碍。

当然，深度学习的数据归一化和经典机器学习的归一化有较大差别，但本质上理论是相通的，本节先从经典机器学习的归一化算法开始讨论，再逐步过渡到深度学习的数据归一化方法。

1）应用差异：传统机器学习归一化是可选择的（有的用，有的不用），而深度学习归一化可谓一个通用的优化算法。

2）度学习的数据归一化和经典机器学习的归一化有较大差别，但本质上理论是相通的

### 1.1 经典机器学习的归一化算法

经典机器学习领域的数据归一化算法主要有两种，分别是0-1标准化（Max-Min Normalization）和Z-Score标准化。二者使用效果差别不大，并且都是逐列对输入的数据进行处理。

#### 1.1.1 0-1标准化方法

0-1标准化是最简单也是最容易想到的方法，同时也是经典机器学习领域最常用的标准化方法。该方法通过在输入特征中逐列遍历其中里的每一个数据，将Max和Min的记录下来，并通过Max-Min作为基数（即Min=0，Max=1）进行数据的归一化处理，基本公式为：
$$
{x}_{normalization}=\frac{x-Min}{Max-Min}
$$
也就是每一列中的元素减去当前列的最小值，再除以该列的极差。 放缩到了0-1区间内。

```python
t = torch.arange(12).reshape(6, 2).float()
t_max = t.max(0)[0]
t_min = t.min(0)[0]
(t - t_min) / (t_max - t_min) # 计算0-1标准化后结果
```

不过在深度学习领域，我们更希望输入模型的数据是Zero-Centered Data，此时Z-Score标准化会更加合适。

#### 1.1.2 Z-Score标准化

和0-1标准化不同，Z-score标准化利用原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。同样是逐列进行操作，每一条数据都减去当前列的均值再除以当前列的标准差。很明显，通过这种方法处理之后的数据是典型的**Zero-Centered Data**，并且如果原数据服从正态分布，通过Z-Score处理之后将服从标准正态分布。Z-Score标准化计算公式如下：
$$
{x}_{normalization}=\frac{x-\mu }{\sigma }
$$

```python
(t - t.mean(0)) / t.std(0)
```

```python
def Z_ScoreNormalization(data):
 stdDf = data.std(0)
 meanDf = data.mean(0)
 normSet = (data - meanDf) / stdDf
 return normSet
```

一种更加严谨的做法，是在分母项、也就是标准差上加上一个非常小的常数𝜇，从而使得**分母恒大于0**。

### 1.2 经典机器学习归一化算法在深度学习中的实践

- 在训练集上训练，测试集上测试

在建模之前，首先需要明确两个问题，其一是标签是否需要标准化（如果是回归类问题的话），其二是测试集的特征是否需要标准化？

1) 首先，**标签是否标准化对建模没有影响**，因此一般我们**不会对标签进行标准化**；

2) 其次，在实际模型训练过程中，由于数据集要划分成训练集和测试集，因此一般来说我们**会在训练集的特征中逐行计算其均值和标准差**，然后进行模型训练，当输入测试集进行测试时，我们会**将在训练集上计算得出的每一列的均值和标准差带入测试集并对测试集特征进行标准化**，然后再带入进行模型测试。值得注意的是，**此时进行标准化时涉及到的每一列的均值和方差也相当于是模型参数**，**必须从训练集上得出，不能借助测试集的数据。**

- Z-Score建模实验

简化了在训练集上计算均值方差再带入测试集进行操作的流程，直接采用全部数据集进行数据归一化操作。

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建最高项为2的多项式回归数据集
features, labels = tensorGenReg(w=[2, -1], bias=False, deg=2)
features_norm = Z_ScoreNormalization(features)

# 进行数据集切分与加载
# labels相同，则可以用同一个test_loader对象接收
train_loader, test_loader = split_loader(features, labels)
train_loader_norm, test_loader = split_loader(features_norm, labels)
```

```python
# 设置随机数种子
torch.manual_seed(24)

# 关键参数
lr = 0.03
num_epochs = 40

# 实例化模型
sigmoid_model3 = Sigmoid_class3()
sigmoid_model3_norm = Sigmoid_class3()

# 进行Xavier初始化
for m in sigmoid_model3.modules():
    if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)

for m in sigmoid_model3_norm.modules():
    if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)

# sigmoid_model3模型训练
train_l, test_l = model_train_test(sigmoid_model3,
 train_loader,
 test_loader,
 num_epochs = num_epochs,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)

# sigmoid_model3_norm模型训练
train_l_norm, test_l_norm = model_train_test(sigmoid_model3_norm,
 train_loader_norm,
 test_loader,
 num_epochs = num_epochs,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)
```

然后对比两种模型的运算结果 ：

```python
plt.plot(list(range(num_epochs)), train_l, label='train_mse')
plt.plot(list(range(num_epochs)), train_l_norm, label='train_norm_mse')
plt.legend(loc = 1)
```

![1691591343772]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_001.png' | relative_url }})

```python
plt.plot(list(range(num_epochs)), test_l, label='test_mse')
plt.plot(list(range(num_epochs)), test_l_norm, label='test_norm_mse')
plt.legend(loc = 1)
```

![1691591368093]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_002.png' | relative_url }})

> 值得注意的是，此处我们统一先对模型进行Xavier参数初始化计算然后再使用数据归一化方法，是为了避免不同随机数种子对参数初始化取值的影响，但实际上目前**很多神经网络用到的归一化方法**（后面会谈到）在**最终效果上都能让模型摆脱参数初始化的限制**，也就是能够让模型在不使用初始化方法的情况下仍然可以进行快速和稳定的收敛。

经过Z-Score归一化的数据收敛速度更快，在某些情况下也能获得更好的结果，当然我们也能通过微观层面、通过查看各层梯度值来观察数据归一化的效果，验证归一化是否有助于各层梯度保持平稳。

```python
# 设置随机数种子
torch.manual_seed(24)

# 关键参数
lr = 0.03
num_epochs = 5

# 实例化模型
sigmoid_model3 = Sigmoid_class3()
sigmoid_model3_norm = Sigmoid_class3()

# 进行Xavier初始化
for m in sigmoid_model3.modules():
    if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)

for m in sigmoid_model3_norm.modules():
    if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)

# sigmoid_model2模型训练
train_l, test_l = model_train_test(sigmoid_model3,
 train_loader,
 test_loader,
 num_epochs = num_epochs,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)

# sigmoid_model2_norm模型训练
train_l_norm, test_l_norm = model_train_test(sigmoid_model3_norm,
 train_loader_norm,
 test_loader,
 num_epochs = num_epochs,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)
```

```python
weights_vp(sigmoid_model3, att="grad")
```

![1691591690117]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_003.png' | relative_url }})

```python
weights_vp(sigmoid_model3_norm, att="grad")
```

![1691591699696]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_004.png' | relative_url }})

由此能看出，使用归一化后的数据进行建模的模型，前几轮迭代时梯度相对平稳。我们知道，当**各层梯度更加均衡时模型学习能力更强，进而能够加快模型收敛速度**，甚至提升模型准确率。因此我们也可以推断，归一化是通过维持梯度平稳来加快收敛速度、提升模型准确率的。

### 1.3 Z-Score数据归一化的局限

#### 1.3.1 Zero-Centered特性消失

尽管Z-Score归一化能够一定程度保证梯度平稳，进而提升模型收敛速度甚至是提升模型效果，但是，和Xavier初始化方法一样，由于是对于“初始值”的修改，因此也会存在**随着迭代次数增加**就逐渐破坏了Zero-Centered Data这一条件的问题，当然，该问题也可视作经典机器学习归一化方法应用于深度神经网络时的局限。

并且，随着参数和输入数据都回到不可控状态，各层的梯度又将回到不可控的状态，而所谓的控制梯度平稳性也将无从谈起。

对于tanh优化迭代5次：

![1691592294094]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_005.png' | relative_url }})

当迭代40次，

![1691592314268]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_006.png' | relative_url }})

能够看出，刚开始时梯度较为平稳，而**迭代到后期**时就出现了明显的**梯度爆炸现象**。

#### 1.3.2 Zero-Centered Data的作用局限

除了输入数据在迭代过程中会逐渐丧失Zero-Centered特性外，Z-Score标准化在应用到深度学习模型中，还会面临一个更加严峻的问题——那就是Zero-Centered Data本身作用范围也是有限的。我们哪怕能够维持输入数据的Zero-Centered特性，也很难保证只凭借这一点就能确保梯度平稳。

而所谓的梯度平稳，我们可以用Glorot条件来进行描述，即正向传播过程中数据流经各层时输入和输出数据方差相同，反向传播时数据流经各层前后该层梯度的方差相同。
$$
\begin{aligned}grad_1&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot w_2\cdot f(X*w_1)\cdot X\\\\grad_2&=\frac{\partial loss}{\partial\hat{y}}\cdot w_3\cdot f(F(X*w_1)*w_2)\cdot F(X*w_1)\\\\grad_3&=\frac{\partial loss}{\partial\hat{y}}\cdot F(F(X*w_1)*w_2)\end{aligned}
$$
由于各层的梯度实际上受到激活函数、各层输入数据和参数三者共同影响，因此哪怕我们将所有的输入数据都调整为零均值的，各层梯度的计算结果还是有可能因为受到其他因素影响导致不平稳。**因此，一味追求输入数据的Zero-Centered或许并不是最好的选择**。

### 1.4 保证梯度平稳的第二条道路：输入数据调整

尽管Z-Score作用有限，但对输入数据进行有效处理，却是至关重要的深度学习模型优化方向。

根据上述各层梯度计算公式，不难发现，影响梯度平稳性的核心因素有三个，其一是各层的参数、其二是各线性层接收到的数据、其三则是激活函数。关于参数的优化我们在Lesson 13中已经进行了详细的介绍，简单来说就是通过Glorot条件巧妙设置参数初始值，从而使得各层梯度在计算过程中尽可能更加平稳。但由于参数本身的特殊性，我们只能设置其初始值，一旦模型开始迭代，参数就将开始“不受控制”的调整，初始值的设置是很难长期保证梯度平稳的，这点和Z-Score对数据进行初始化所存在的问题一致。
除了参数调整外，在确保梯度平稳性上我们就只剩下选择激活函数和调整输入数据两条路可走。关于新兴激活函数的选择我们将在下一节课进行详细讨论，但对于输入数据的优化，目前来看，应用最为广泛、并且被验证的实践效果最好的数据归一化方法，是由Sergey loffe和Christian Szegedy在2015年发表的[《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》](http://de.arxiv.org/pdf/1502.03167)中提出的一种方法，**该方法通过修改每一次带入训练的数据分布（每一个Batch）的数据分布，来提升模型各层梯度的平稳性，从而提升模型学习效率、提高模型训练结果**。

由于是修改每一个Batch的数据分布，因此该方法也被称为 Batch Normalization（BN），小批量数据归一化方法。不过需要注意的是，尽管BN也是一种数据归一化方法，但深度学习领域的归一化方法和经典机器学习的归一化方法却存在很大区别，经典机器学习的归一化方法主要是为了消除不同特征的量纲影响而对各列的数据分布进行修改，并且并非所有机器学习模型都要采用数据归一化方法才能进行有效建模，而深度学习归一化方法的目标则是为了确保模型能够进行有效训练为目的，是一种适用于所有模型的一种必要的优化方法。

### 1.5 Batch Normalization基础理论

尽管目前来看，Batch Normalization 已经是被验证的、行之有效的模型优化手段，但BN的诞生及其有效性的原理证明，却有一段有趣的历史。

根据《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》一文中的观点，作者认为在深度神经网络模型训练过程中，容易出现一种名为内部协方差偏移（Internal Covariate Shift、ICS）的问题，该问题会导致模型性能下降，而为了解决该问题，作者提出了BN方法，并在各种实验中验证了BN对模型效果提升的有效性。

然而，到底什么是ICS，作者在原文中只给出了简单的描述而未给出严格定义，并且对于BN可以提升模型性能方面，作者表示根本原因是因为BN能够消除ICS。有趣的是，尽管这套理论没有在原文中给出更加严谨的说明和论证，但由于BN方法本身在应用实践中效果很好，因此大多数情况下人们并未对原文中论述不严谨的理论推导部分进行深究。直到2018年，来自MIT的研究团队发表论文，[《How Does Batch Normalization Help Optimization?》](https://arxiv.org/pdf/1805.11604.pdf)，研究人员表示，通过一些列严谨的实验可以证明，BN方法对模型优化的有效性和原论文中所描述的消除ICS没有任何关系，甚至在某些场景下BN还会加剧模型的ICS，而到底是什么让BN如此有效，MIT的研究团队最终也没有给出严谨的理论证明。如此一来BN方法就成了基础理论“不成立”但实际上很好用的一项优化方法，而对于BN的有效性的理论研究，也成为了诸多科研团队的命题。

不过，尽管BN有效性原理成谜，但MIT的研究团队通过一系列的实验，最终还是给出了一些BN有效性的理论推断，他们判断BN之所以有效，**应该是和修改数据分布、从而使得梯度更加平稳有关，也就是该方法之所以能够发挥作用，仍然可能和满足了Glorot条件有关**。因此，在介绍BN方法时，我们将不再讨论BN是否能够ICS这一命题，转而探讨是BN如何修正数据分布进而使得模型能够更好满足Glorot条件这一点入手，尽可能帮助学员建立对BN方法有效性的判别依据。

#### 1.5.1 归一化方法与数据分布的相互独立性

**任何归一化的本质都是对数据进行平移和放缩（自适应跳调整过程）**，所谓平移，就是指数据集每一列统一加上或减去某一个数，在Z-Score中就是每一列减去该列的均值，而所谓的放缩，就是指数据集中每一列数据统一除以或乘以某一个数，在Z-Score中就是每一列除以当前列的标准差。而数据的平移和放缩，**是不会影响数据特征的分布情况的。 也就是不影响原始数据的规律。**

![1691595536008]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_007.png' | relative_url }})

而比较重要的一点是，数据的分布其实就代表着数据背后的规律，我们使用模型去捕捉数据规律，其实就是对数据分布情况进行学习。因此，数**据归一化不修改数据分布，是我们使用归一化方法的基本前提**，否则，一旦数据归一化方法会修改数据分布，则相当于是人为破坏了数据原始规律，这将会对后续的模型学习造成巨大的影响。

#### 1.5.2 归一化与仿射变换

归一化的本质仍然是对数据进行仿射变换。根据此前介绍，数据的仿射变换用矩阵形式来进行表示就是：
$$
\hat x = x * w + b
$$
其中x是原数据，w是参数矩阵，b是截距，𝑥̂ 是经过变换之后的数据。此处以Z-Score为例，探讨归一化操作转化为仿射变换的方法。

在归一化运算时，我们进行了如下操作：
$$
\frac{x-mean(x)}{std(x)}
$$
当然，该过程也可以写成如下表达式：
$$
\frac{x}{std(x)} - \frac{mean(x)}{std(x)} = x \cdot\frac{1}{std(x)}- \frac{mean(x)}{std(x)} = x \otimes \gamma + \beta
$$
其中⊗表示逐个元素相乘，𝛾和𝛽为指代参数

```python

# 创建一组数据
f = torch.arange(1, 5).reshape(2, 2).float()
r = torch.tensor(1.) / f.std(0) # 每列的标准差
```

而原数据每列的标准差分之一和原始数据的每一列相乘，可以通过如下矩阵相乘形式计算得出：

torch.diag(r) 构建以每列标准差分之一为对角线元素的对角矩阵

```python
rm = torch.diag(r) # 构建以每列标准差分之一为对角线元素的对角矩阵
torch.mm(f, rm) # 计算得出每一列除以标准差之后的结果
b = -(f.mean(0) / f.std(0))

torch.mm(f, rm) + b
'''tensor([[-0.7071, -0.7071],
 [ 0.7071, 0.7071]])'''
Z_ScoreNormalization(f)
'''tensor([[-0.7071, -0.7071],
 [ 0.7071, 0.7071]])'''
```

至此，我们可以发现，在归一化的过程中，放缩的部分可以通过矩阵乘法来完成，假设原数据为一个m*n的数据集，则令其左乘一个对角线元素为原数据集每一列放缩参数的矩阵即可完成放缩，而平移部分则更加简单，令数据集加上由每一列平移幅度所构成的向量即可完成平移。

而如果归一化是一种放射变换，并且归一化的计算过程可以用矩阵运算来完成，那么就有两方面的直接影响：

- 其一，PyTorch 中对数据的仿射变换是用线性层完成的，既然归一化是一种特殊的仿射变换，归一化的操作也可以由一个特殊的线性层来完成，并且和所有的线性层一样，归一化层也可以放置在任何隐藏层前后。

- 其二，在具体实现层面上，在仿射变换中我们是可以设置参数、并且通过构建损失函数再利用梯度下降进行求解的，那么对于归一化这种特殊的仿射变换，平移和放缩的尺度是否可以简单设计成一个参数，然后带入模型进行训练求解呢?

如前文所述，将数据转化为Zero-Centered Data并非最好的选择，如果可以通过梯度下降求解出最佳放缩和平移的尺度，大概率会比简单使用Z-Score将其放缩为均值为0、方差为1的数据分布效果更好（至少不会差）

#### 1.5.3 Batch Normalization基本流程

作为针对数据归一化方法，Batch Normalization 的理论实现流程看似更加复杂，但实际上，和所有的归一化方法一样，**BN做的也就是对输入数据进行平移和放缩**。只不过BN的平移和放缩分为两个阶段，第一个阶段是进行Z-Score处理，也就是将输入数据的均值平移至0点，并将输入数据的方差放缩至1；第二阶段则是在此基础上在对数据的均值进行参数平移（加上一个通过训练、也就是梯度下降算法算得的参数），并对其方差进行参数（另一个通过模型训练、也就是梯度下降算得的参数）放缩。具体数学过程如下：

Stage 1：Z-Score放缩
$$
\hat{x}^{(i)} = \frac{x^{(i)}-\mu_B}{\sqrt{\delta^2_B+\epsilon}}
$$
&emsp;&emsp;其中$\mu_B$代表输入的小批数据均值，$\delta^2_B$代表输入的小批数据方差，$\hat{x}$代表Z-Score归一化后的数据结果，和此前一样，Z-Score计算过程为减去每一列的元素减去该列的均值除以该列的修正后的标准差（方差加上一个极小的数使得其不为零）。第一阶段的参数都是可以计算或者人工设置的，其中均值和方差都是简单计算即可得出，而修正项$\epsilon$则是可以手动设置的。

Stage 2：带参数的平移和放缩
$$
z^{(i)}=\gamma \otimes \hat{x}^{(i)}+\beta
$$
&emsp;&emsp;和第一阶段参数可以简单计算得出不同，二阶段的参数，也就是$\gamma$和$\beta$都是模型参数，在实际模型训练过程中是需要作为模型整体参数的一部分，带入损失函数、进而通过梯度下降计算得出的。而二阶段的实际处理数据过程，则是首先对一阶段放缩之后**每一列的$\hat{x}$乘以一个固定的$\gamma$在加上一个固定的$\beta$。**需要强调的是，上述公式是针对列进行处理，即每一列都乘以一个$\gamma$再加上一个$\beta$，并不是所有数据都乘以一个$\gamma$再加上一个$\beta$，也不是每一个数据都乘以一个$\gamma$再加上一个$\beta$。

实际上**BN的第一个阶段和第二个阶段都是在做同一件事——都是在对数据进行平移和放缩**，甚至当$\gamma=\sqrt{\delta^2_B+\epsilon}, \beta=\mu_B$，二阶段处理结果将还原回原始输入数据，也就是$z^{(i)}=x^{(i)}$

既然两个阶段做的相同的事情，为何不能直接合并为一个阶段。从理论上来说，我们可以将上述过程合并为：
$$
\begin{aligned}
z^{(i)}
& = \gamma \otimes \hat{x}^{(i)}+\beta \\
& = \gamma \otimes \frac{x^{(i)}-\mu_B}{\sqrt{\delta^2_B+\epsilon}} + \beta \\
& = \frac{\gamma}{\sqrt{\delta^2_B+\epsilon}}\otimes x^{(i)}+(\beta-\frac{\gamma \cdot \mu_B}{\sqrt{\delta^2_B+\epsilon}}) \\
& = \hat\gamma \otimes {x}^{(i)}+\hat \beta \\
\end{aligned}
$$
实际 PyTorch 模型训练完成后输出的参数结果也是原始的$\gamma和\beta$。这里我们可以这样理解：BN方法无疑会增加模型计算量，而第一阶段**提前对数据进行Z-Score处理（调整为0均值，1方差分布）**，就相当于是先选择了一组初始的$\gamma和\beta$对数据进行处理（此时$\gamma_0=\frac{1}{\sqrt{\delta^2_B+\epsilon}}, \beta_0 = \frac{-\mu_B}{\sqrt{\delta^2_B+\epsilon}}$），然后再此基础之上进行带参数的平移和缩放，二阶段平移和缩放的参数是需要经过梯度下降迭代计算得出，而选择在$\gamma_0和\beta_0$基础上进行梯度下降计算，就相当于选择了一组初始值点。

分两步意义：

- $\gamma_0和\beta_0$该初始值点能够减轻计算量（对整体大量数据求均置于方差会消耗大量算力），节省计算资源，并有效提升迭代收敛速度。

- 后面测试集上测试时，不能带入小批量的均值和方差，通过算到整体均值和方差则可以带入。

并且我们需要知道两种特殊情况：

- 其一，当$\gamma = 1, \beta=0$时，上述仿射变换过程就退化为Z-Score变换；
- 其二，当$\gamma=\sqrt{\delta^2_B+\epsilon}, \beta=\mu_B$时，就相当于退回原始数据集。

#### 1.5.4 Batch Normalization方法有效性认知

&emsp;&emsp;接下来，我们从理论角度，简单分析BN方法的实际作用。我们知道，一旦$\gamma和\beta$训练完成、也就是数值确定，经过**BN处理完的数据将朝向方差为$\gamma$、均值为$\beta$的分布方向靠拢**。也就是**BN作为一种特殊的仿射变换，会调整经过BN层的数据分布，而数据分布的均值和方差又是由模型训练得到的，并不一定像Z-Score一样是调整为0均值1方差的数据分布，因此我们可以理解为BN在调整数据分布时其实是朝着最有利于提升模型效果的方向调整**，或者说，朝着满足损失函数能取得更小数值的方向调整。在这个过程中，调整后的数据分布可以以任何形态出现，并不局限于0均值和1方差的数据分布，这一点是BN方法和Z-Score方法最核心的区别，当然，如果我们反向论证，**如果最终模型算得最佳分布不是0均值1方差的分布，那么也就说明Z-Score归一化方法并不是适用于当前模型的最佳归一化方法**。
&emsp;&emsp;另外，我们前面也讨论过，对数据进行归一化处理，也就是放缩和平移，其实是不会改变数据分布规律的，因此我们可以在模型训练的任何时候、任意阶段对数据进行放缩和平移。而BN方法本身又是一种数据分布自适应的方法，可以根据实际情况训练出最合适的数据分布，因此我们可以将BN方法放置于神经网络的各个线性层前后的，协助模型即时调整数据在每一层的分布。当然，就像此前说的一样，其实我们也可以对每一层数据进行Z-Score处理，但可想而知，效果应当不如数据分布自适应的BN方法好。
&emsp;&emsp;如果这种**自适应的数据分布调整是朝向损失函数取值更小的方向调整，并且数据分布直接影响的就是各线性层的梯度平稳行，而梯度平稳又是保证模型有效性的根本原因之**一，据此我们或许也可以从理论上推导出这种数据分布上的自适应的调整确实是朝向保证模型梯度平稳的方向调整的。

## 2 Batch Normalization在PyTorch中的实现

### 2.1 nn.BatchNorm基本使用方法

#### 2.1.1 nn.BatchNorm类介绍

在 PyTorch 中，我们使用nn.Linear构建线性层，类似的，我们通过使用nn.BatchNorm类来构建BN层，进而可以进行 Batch Normalization 归一化操作，并且在后续操作过程中我们会发现，BN层和线性层之间由诸多相似之处。同时，出于对不同类型数据的不同处理需求，nn.BatchNorm1d类主要用于处理2d数据，nn.BatchNorm2d则主要用来处理3d数据。简单来说，目前我们所使用的面板数据（二维表格数据）都属于2d数据，使用nn.BatchNorm1d处理即可。而后续在处理图像数据时，则需要视情况和使用nn.BatchNorm2d类。

```python
nn.BatchNorm1d(
 num_features,
 eps=1e-05,
 momentum=0.1,
 affine=True,
 track_running_stats=True,
)
```

- num_features：输入数据的特征数量（假设为n），也就是前一层神经元数量或原始数据集特征数量，根据此前的论述，BN层最终构建的是一个n*n的对角矩阵，对角线元素包含$\gamma$，并且截距项为$\beta$；
- eps：方差分母修正项，为了防止分母为0，一般取值为1e-5，也就是类默认值；就是下列中的$\epsilon$

$$
\hat{x}^{(i)} = \frac{x^{(i)}-\mu_B}{\sqrt{\delta^2_B+\epsilon}}
$$

- affine：是否进行仿射变换，需要注意的是，此时进行仿射变换时将使用无偏估计进行期望和方差的计算，并且初始条件下$\gamma=1，\beta=0$，当参数取值为True时，**会显式设置$\gamma和\beta$参数并带入进行梯度下降迭代计算**，取值为False时，参数不显示，实际的数据归一化过程就是对原数据进行无偏估计下的Z-Score变换。

BN中采用无偏估计进行方差和均值计算，PyTorch 中默认的方差计算是样本无偏估计，而BN再进行归一化计算时实际上采用的是样本方差，二者区别如下，样本方差为：
$$
Var(x) = \frac{\sum^n_{i=1}(x_i-\bar x)^2}{n}
$$
但对样本整体方差的无偏估计为：
$$
Var_{unbiased}(x) = \frac{\sum^n_{i=1}(x_i-\bar x)^2}{n-1}
$$

实例化一个BN类，需要输入数据的特征数量（假设为n）

```python
# 实例化一个BN类
bn1 = nn.BatchNorm1d(3)
```

```python
# 整体方差的无偏估计
torch.var(t)
# 样本方差
torch.var(t, unbiased=False)
```

- .parameters() 查看BN层的参数情况

```python
list(bn1.parameters())
'''[Parameter containing:
 tensor([1., 1., 1.], requires_grad=True),
 Parameter containing:
 tensor([0., 0., 0.], requires_grad=True)]'''
```

据此也可确认在迭代开始之前，$\gamma和\beta$的初始值。同时，根据参数的可微性，我们也能看出其最终也是需要通过反向传播计算梯度，然后利用梯度下降进行求解的。当然，此处如果我们将affine参数设置为False，则无法查看参数，并且归一化过程就是简单的在无偏估计下对数据进行Z-Score归一化。

- 训练数据均值与方差的迭代计算

对于BN层来说，计算过程并不复杂，但有一个**核心需要解决的问题**：在测试阶段、针对可能出现的一条条测试集输入的情况，应该如何对测试数据进行归一化处理。

针对此问题，BN层采用了机器学习一以贯之的基本思路，那就是所有用于处理测试数据的参数都在训练数据中得出，包括**归一化所需要用到的均值和方差**，并且**对于测试集在归一化过程中所用到的均值和方差，都是训练数据整体均值和方差**。但是，在模型训练阶段，数据是分一个个小批输入的，而针对小批训练数据的归一化处理也没有采用所有训练数据的均值和方差，而是采用训练数据总体均值和方差的无偏估计带入进行计算。

值得注意的是，**针对测试集计算的均值和方差就是真实计算出的训练数据的均值和方差**，而不是一个估计的结果。

为了能够获取训练数据整体均值和方差，BN采用了迭代累计计算的方法，通过类似动量法的方法对每一个小批数据的均值和方差进行累计统计，最终计算训练数据整体均值和方差，具体过程如下。

BN层在每一次进行归一化时，都会通过“某种”方法累计输入数据集的均值和方差，在每一次处理完一批数据之后，我们都可以通过调用BN层的running_mean和running_var查看BN层所累计的均值和方差结果：

```python
bn1.running_mean
bn1.running_var
'''tensor([0.3000, 0.4000, 0.5000])
tensor([1.8000, 1.8000, 1.8000])'''
```

不过值得注意的是，BN层并不是记录输入数据集的均值和方差，而是采用了一种类似动量法的方法在累计每一个小批数据的统计量。这种累计过程如下：首先，当我们每次实例化一个BN层之后，都会获得一组原始的running_mean和running_var

```python
bn2 = nn.BatchNorm1d(3)
bn2.running_mean
bn2.running_var
'''tensor([0., 0., 0.])
tensor([1., 1., 1.])'''
```

初始条件下，running_mean取值为0，running_var取值为1。而当我们在使用BN层进行每一次归一化数据时，BN层的running_mean和running_var将按照如下公式进行调整：
$$
running\_mean = (1-momentum) * running\_mean + momentum * sample\_mean
$$

$$
running\_var = (1-momentum) * running\_var + momentum * sample\_var
$$

其中，𝑚𝑜𝑚𝑒𝑛𝑡𝑢𝑚是BN层实例化过程中的可选参数，默认值为0.1，sample表示当前输入的小批数据。上述过程可简单理解为BN层每每次归一化一批数据，都会针对围绕已经记录的running_mean/var和小批数据的sample_mean/var(样本均值与方差)进行加权求和，在momentum取值为0.1的情况下，原running_mean/var权重是sample_mean/var的九倍，也就是说，当前数据归一化完成后，running_mean/var会朝向sample_mean/var小幅移动。

能够发现，BN记录结果朝向整体真实结果靠拢。

- 迭代记录的原因

至于为什么神经网络为什么采用迭代方法求解训练集特征的均值和方差，而不是直接直接按列进行统计。BN通过迭代求解的主要原因有两点，其一是面对超大规模数据时候求解均值或方差可能无法直接进行求解，尤其是如果数据是分布式存储的话一次性求解均值和方差就会更加困难；其二，迭代计算是深度学习求解参数的基本过程，所有深度学习框架为了更好地满足迭代计算，都进行了许多精巧的设计，从而使得迭代计算更加高效和平稳，而在计算训练集整体均值时，如果借助神经网络向前传播的过程进行迭代记录，便能够在控制边际成本的情况下快速达成目标。

- momentum参数设置

关于momentum的设置，一般来说，为了尽可能获取到更加准确的训练集整体统计量，当每一个小批数据数据量比较小时，我们应该将历史数据比重调高，也就是降低momentum取值，以减少局部规律对获取总体规律的影响，当然，此时我们也需要增加遍历数据的次数epochs；而反之则可以考虑增大momentum取值。

- track_running_stats参数

对于BN来说，最后一个参数就是track_running_stats，默认取值为True。当此参数为True时，BN层会在每次迭代过程中会结合历史记录更新running_mean/var，当track_running_stats=False时，BN层将running_mean/var为None，并且在进行预测时会根据输入的小批数据进行均值和方差计算。在大多数情况下，并不推荐修改该参数的默认取值。

注意，迭代过程需要设置随机数种子，否则每一轮迭代都将创建不同的小批数据，也将影响最终BN层记录结果：

#### 2.1.2 model.train()与model.eval()

- 伴随向前传播调整running_mean/var

尽管BN找到了能够计算训练集整体均值和方差的方法，但截至目前，还是没有解决如何将在训练过程中记录的running_mean/var应用到测试集当中的方法。核心原因在于，B**N层的running_mean/var是伴随向前传播同步调整的**。也就是说，**在track_running_stats开启时，只要进行一次向前传播，就会更新一次running_mean/var**，这明显是不合适的（**因为我们不能根据测试集数据修改参数**）。但是我们也知道，只要要进行测试集的测试，就一定需要模型执行测试数据的向前传播以得出模型结果，此时应该怎么办呢？

> 此前我们没有遇到该难题的主要原因是，线性层的参数调整是通过optimizer.step()完成的，也就是说我们需要手动输入代码才能完成线性层参数调整，在模型训练过程，我们每一次迭代都会手动调整一次线性层参数，而在输入测试数据时我们只会执行向前传播，并不会在向前传播结束后进行optimizer.step()，因此输入测试数据不会影响模型参数调整。

- model.train()与model.eval()用于提供两种不同模式

此时就要用到 PyTorch 中适用于nn模块中所有模型的一种方法——model.train()与model.eval()。其中，**model.train()表示开启模型训练模式**，在默认情况下，我们实例化的每一个模型都是出于训练模式的，而**model.eval()则表示将模型转化为测试模式**。我们可以简单查看该方法的相关说明。

那么，训练模式和测试模式的根本区别是什么？简单来说就是，在 PyTorch 中，其实有很多类似BN类一样，**会在向前传播过程中直接自动修改模型参数，而当模型出于测试模式时，会避免这种情况发生**，即模型出于测试模式时不会根据当前输入数据情况调整running_mean/var。并且，模型出于测试模式时，BN层会利用已经记录下的running_mean/var对数据进行归一化。

```python
#实例化BatchNorm1d
bn5 = nn.BatchNorm1d(3)
bn5.train()
'''BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)'''
bn5.train(False) #等价于 bn5.eval()
```

### 2.2 利用nn.BatchNorm构建带BN的神经网络

有了对nn.BatchNorm类基本了解之后，我们来尝试构建带有BN层的神经网络模型。根据上一节的介绍，**我们知道BN本质上是一种自适应的数据分布调整算法，同时我们也知道，数据分布的调整并不影响，因此我们可以在任何需要的位置都进行BN归一化**。另外，**根据Glorot条件，我们知道在模型构建过程中需要力求各梯度计算的有效性和平稳性，因此我们可以考虑在每一个线性层前面或者后面进行数据归一化处理**。

在实际模型构建过程中，**在模型中添加BN层和添加线性层类似，我们只需要在自定义模型类的init方法中添加BN层，并在向前传播方法中确定BN层的调用方法即可**。而在具体BN层位置选择方面，是放在线性层前面还是放在线性层后面，目前业内没有定论，既没有理论论证BN层的最佳位置，也没有严谨的实验证明在哪个位置放置BN层效果最好。因此我们大可根据自己的习惯，以及自身经验的判断选择BN层的位置。当然，为了实验方便，我们会同时创建可以前置或者后置BN层的模型。（这里我们将放在隐藏层前面（也就是线性层后面，**激活函数前面**）称为**BN层前置**，放在隐藏层后面，**激活函数后面**称为**BN层后置**。）

#### 2.2.1 构建模型类

们创建一个同时能够调整激活函数、可选是否包含BN层、以及BN层放置位置的模型，当然，该模型仍然是分层创建，在包含了上述参数后相当于此前模型的一个集成。我们先从简单入手，创建一个两层神经网络、BN层可选、激活函数可选的模型。

```python
class net_class1(nn.Module):
    def __init__(self, act_fun=torch.relu, in_features=2, n_hidden=4, out_features=1, bias=True, BN_model=None, momentum=0.1):
        super(net_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun

 def forward(self, x):
    # 不带BN层
    if self.BN_model None:
    z1 = self.linear1(x)
    p1 = self.act_fun(z1)
    out = self.linear2(p1)
    # BN层前置
    elif self.BN_model 'pre':
    z1 = self.normalize1(self.linear1(x))
    p1 = self.act_fun(z1)
    out = self.linear2(p1)
    elif self.BN_model 'post':
    z1 = self.linear1(x)
    p1 = self.act_fun(z1)
    out = self.linear2(self.normalize1(p1))
    return out
```

```python
# 设置随机数种子
torch.manual_seed(420)

# 创建最高项为2的多项式回归数据集
features, labels = tensorGenReg(w=[2, -1], bias=False, deg=2)

# 进行数据集切分与加载
train_loader, test_loader = split_loader(features, labels)

# 实例化一个前置BN层的神经网络
torch.manual_seed(24)
relu_model1_norm = net_class1(BN_model='pre')

# 设置模型为训练模式
relu_model1_norm.train()

```

这里我们只需要对模型整体进行训练模式设置relu_model1_norm.train()，模型中所有模块都将调整为训练模式。

```python
list(relu_model1_norm.modules())
'''[net_class1(
 (linear1): Linear(in_features=2, out_features=4, bias=True)
 (normalize1): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 (linear2): Linear(in_features=4, out_features=1, bias=True)
 ),
 Linear(in_features=2, out_features=4, bias=True),
 BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 Linear(in_features=4, out_features=1, bias=True)]'''
for m in list(relu_model1_norm.modules()):
    print(m.training True)
'''True
True
True
True'''
```

值得注意的是，此时BN层记录的均值和方差都有4个分量，而原始训练数据只有两个特征。本质上是因为此时BN层是围绕第一个线性层的输出结果进行归一化处理，而数据经过第一个线性层时将转化为N*4的形状。

```python
list(relu_model1_norm.modules())[2].running_mean
list(relu_model1_norm.modules())[2].running_var
'''tensor([0., 0., 0., 0.])
tensor([1., 1., 1., 1.])'''
```

注意，当我们调用BN层的模块来查看参数时，weight就是𝛾，bias就是𝛽。

```python
list(relu_model1_norm.modules())[2].weight
'''Parameter containing:
tensor([1., 1., 1., 1.], requires_grad=True)'''
list(relu_model1_norm.modules())[2].bias
'''list(relu_model1_norm.modules())[2].bias'''
```

我们发现BN层确实得到了训练。当然，我们也可以查看BN层记录的统计指标

```python
list(relu_model1_norm.modules())[2].running_mean
list(relu_model1_norm.modules())[2].running_var
'''tensor([-0.1037, -0.2748, -0.1374, 0.0303])
tensor([4.2056, 5.7263, 6.9815, 4.9587])'''
```

假设h（x）是一个激活函数。

- 1. 饱和

1. 当我们的x趋近于正无穷，h（x）'趋近于0，那么我们称之为右饱和。
2. 当我们的n趋近于负无穷，h（x）'趋近于0，那么我们称之为左饱和。
3. 当一个函数既满足左饱和又满足右饱和的时候我们就称之为饱和，典型的函数有Sigmoid，Tanh函数。

- 2. 硬饱和

1. 对于任意的x，如果存在常数c，当x>c时，恒有h（x）’=0，则称其为右硬饱和。
2. 如果对于任意的x，如果存在常数c，当x<c时，恒有h（x）’=0,则称其为左硬饱和。
3. 既满足左硬饱和又满足右硬饱和的我们称这种函数为硬饱和。

3. 软饱和

1. 对于任意的x，如果存在常数c，当x>c时，恒有h（x）'趋近于0，则称其为右软饱和。
2. 如果对于任意的x，如果存在常数c，当x<c时，恒有h（x）'趋近于0,则称其为左软饱和。
3. 既满足左软饱和又满足右软饱和的我们称这种函数为软饱和。

- 4. 不满足饱和函数条件的函数则称为非饱和激活函数：

ReLU及其变体则是“非饱和激活函数”。

使用“非饱和激活函数”的优势在于两点：

- "非饱和激活函数”能解决所谓的“梯度消失”问题。
- 它能加快收敛速度。

**ReLU激活函数的提出就是为了解决梯度消失问题。**ReLU的梯度只可以取两个值：0或1，当输入小于0时，梯度为0；当输入大于0时，梯度为1。好处就是：ReLU的梯度的连乘不会收敛到0，连乘的结果也只可以取两个值：0或1 。如果值为1，梯度保持值不变进行前向传播；如果值为0 ,梯度从该位置停止前向传播。

我们发现，加入BN层之后，模型效果反而不如原始模型。不难发现，BN优化算法使用起来其实并不简单，并不是只要加入BN层模型效果就一定会有所提升。要充分发挥BN层的效果，需要结合其他模型参数进行综合调参

## 3 Batch Normalization综合调参实战

### 3.1 Batch Normalization与Batch_size综合调参

BN是一种在长期实践中被证明行之有效的优化方法，但在使用过程中首先需要知道，BN的理论基础（尽管不完全正确）是以BN层能够有效预估输入数据整体均值和方差为前提的，如果不能尽可能的从每次输入的小批数据中更准确的估计整体统计量，则后续的平移和放缩也将是有偏的。而由小批数据估计整体统计量的可信度其实是和小批数据本身数量相关的，如果小批数据数量太少，则进行整体统计量估计时就将有较大偏差，此时会影响模型准确率。

因此，一般来说，我们在使用BN时，至少需要保证小批数据量（batch_size)在15-30以上，才能进行相对准确的预估。此处我们适当调整小批数据量参数，再进行模型计算。

我们发现，当提升batch_size之后，带BN层的模型效果有明显提升，相比原始模型，带BN层的模型拥有更快的收敛速度。

当然，为了确保BN层对整体统计量估计的可信度，除了提高batch_size之外，还能够通过调低momentum参数来实现，当然，伴随着momentum值得降低，我们也必须进一步提升遍历数据集得次数，同学们可以根据上述代码自行进行实验。

### 3.2 复杂模型上的Batch_normalization表现

一般来说，**BN方法对于复杂模型和复杂数据会更加有效**，换而言之，很多简单模型是没必要使用BN层（徒增计算量）。对于上述net_class1来说，由于只存在**一个隐藏层，因此也不会存在梯度不平稳的现**象，而BN层的优化效果也并不明显。接下来，我们尝试构建更加复杂的模型，来测试BN层的优化效果。

从另一个角度来说，其实我们是建议更频繁的使用更加复杂的模型并带上BN层的，核心原因在于，复杂模型带上BN层之后会有更大的优化空间。

**BN层对更加复杂模型的优化效果更好**。换而言之，越复杂的模型对于梯度不平稳的问题就越明显，因此BN层在解决该问题后模型效果提升就越明显。

不过呢，和Lesson 13.2中我们看到的一样，模型复杂度提升也是过犹不及的，当模型太过于复杂时，仍然会出现模型效果下降的问题。

对于Sigmoid来说，BN层能很大程度上缓解梯度消失问题，从而提升模型收敛速度，并且小幅提升模型效果。而对于激活函数本身就能输出Zero-Centered结果的tanh函数，BN层的优化效果会更好。

相比Sigmoid，使用**tanh激活函数本身就是更加复杂的一种选择，因此，BN层在tanh上所表现出的更好的优化效果，也能看成是BN在复杂模型上效果有所提升。**

### 3.3 包含BN层的神经网络的学习率优化

BN层对模型迭代的平稳性提升帮助不大，相反，加入BN层的模型收敛过程“不平稳”的特点好像有增无减，这点从Sigmoid激活函数的收敛过程看的尤其明显。

而收敛不平稳的模型，一般都对学习率非常敏感（相关内容我们会在后续学习率优化章节详细讨论），也就是学习率的调整将有效缓解迭代不平稳的问题，而一旦迭代不平稳被修正，模型就有可能最后收敛到一个更优的结果，当然，这只是有可能，最终结果还需要看到底是什么原因导致模型收敛过程不平稳。**而BN层所带来的不平稳性，我们可以简单理解成模型可以在更大范围搜索最优解，相比不带BN层的神经网络模型，带BN层的神经网络的不平稳会更大程度受到学习率的影响**。换而言之，带BN层的神经网络模型对学习率是高度敏感的，并且**带BN层的神经网络模型，在进行学习率调整时能够有更大的优化空间**。也就是说，相比不带BN层的模型，**带BN层的模型在同样进行某种学习率调整时，会有更好的效果**。

为了更好的说明所谓优化空间，我们需要铺垫两个基础认知，其一是学习率敏感度，其二是学习率学习曲线（伴随学习率调整模型效果变化曲线）。

#### 3.3.1 学习率敏感度

#### 3.3.2 学习率学习曲线

学习率作为模型重要参数，学习率的调整也会影响实际模型效果。

伴随着学习率的调整，模型效果如何变化的曲线。

学习率调小之后模型出现这种情况，很大概率是因为学习率较大时，迭代到后期会出现模型迭代解在最小值点附近反复震荡，出于各种原因，无法抵达最小值点。而当我们调小学习率之后，迭代解就能够通过更小的孔。但这也不是绝对的，如果学习率调的过小，不仅会降低迭代速度，同时也极有可能导致迭代解在最小值附近停止不动（每次移动步幅过小）。也就是对于学习率学习曲线来说，实际上也是个U型曲线。我们尝试将学习率调整为0.005和0.001进行建模。

```python
# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model2 = net_class2(act_fun= torch.tanh, in_features=5)
tanh_model2_norm = net_class2(act_fun= torch.tanh, in_features=5, BN_model='pre')

# 创建模型容器
model_l = [tanh_model2, tanh_model2_norm]
name_l = ['tanh_model2', 'tanh_model2_norm']

# 核心参数
lr = 0.001
num_epochs = 40

# 模型训练
train_lss, test_lss = model_comparison(model_l = model_l,
 name_l = name_l,
 train_data = train_loader,
 test_data = test_loader,
 num_epochs = num_epochs,
 criterion = nn.MSELoss(),
 optimizer = optim.SGD,
 lr = lr,
 cla = False,
 eva = mse_cal)

# 学习率为0.01时模型误差
train_lss[1:,-5:].mean()
test_lss[1:,-5:].mean()
```

```python
'''tensor(9.4691)
tensor(16.9594)'''
```

绘图：

```python
lr_l = [0.03, 0.01, 0.005, 0.001]
train_ln = [train_lh[1:,-5:].mean(), train_ls[1:,-5:].mean(), train_lms[1:,-5:].mean(), train_lss[1:,-5:].mean()]
test_ln = [test_lh[1:,-5:].mean(), test_ls[1:,-5:].mean(),
test_lms[1:,-5:].mean(), test

plt.plot(lr_l, train_ln, label='train_mse')
plt.plot(lr_l, test_ln, label='test_mse')
plt.legend(loc = 1)
#plt.ylim(0, 15)
```

![image-20230811232524972]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_008.png' | relative_url }})

对于学习率的调整，一般都会出现倒U型曲线。我们能够发现，在当前模型条件下，学习率为0.005左右时模型效果较好。

对于不包含BN和包含BN：

![image-20230811233215521]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_009.png' | relative_url }})

整体来看，带**BN层的模型对学习率调整更加敏感，优化空间更大**。

BN层这种伴随模型更加复杂、对学习率也更加敏感的属性，最终会使得更加复杂的模型在相同的学习率下表现更好。这是一个隐藏很深的结论，但同时也是一个非常有用的结论。

当然，截至目前，带BN的tanh4在当前数据集上表现最好，并且远超不带BN的其他所有神经网络。据此我们可以得出结论，学习率调整在复杂且带BN层模型上表现效果更好。但值得一提的是，对于加入BN层的tanh模型的调参，目前还只是冰山一角，以目前所掌握的调参手段，尚未发挥BN层的全部效能，因此效果仍然只能说是中规中矩。

#### 3.3.3 带BN层的神经网络模型综合调整策略总结

- 简单数据、简单模型下不用BN层，加入BN层效果并不显著；

- BN层的使用需要保持running_mean和running_var的无偏性，因此需要谨慎调整batch_size；

- 学习率是重要的模型优化的超参数，一般来说学习率学习曲线都是U型曲线；

- 从学习率调整角度出发，对于加入BN层的模型，学习率调整更加有效；对于带BN层模型角度来说，BN层能够帮助模型拓展优化空间，使得很多优化方法都能在原先无效的模型上生效；

- 对于复杂问题，在计算能力能够承担的范围内，应当首先构建带BN层的复杂模型，然后再试图进行优化，就像上文所述，很多优化方法只对带BN层的模型有效；

其他拓展方面结论：

- 关于BN和Xavier/Kaiming方法，一般来说，使用BN层的模型不再会用参数初始化方法，从理论上来看添加BN层能够起到参数初始化的相等效果（参数初始化的作用：保证模型初始迭代过程中各个线性层的梯度平稳性。对输入数据进行一个自适应调整，来达到整体梯度平稳的最好方法，参数则不用初始化调整，并且也耗空间）；（另外，带BN层模型一般也不需要使用Dropout方法）

- 本节尚未讨论ReLU激活函数的优化，相关优化方法将放在后续进行详细讨论，但需要知道的是，对于ReLU叠加的模型来说，加入BN层之后能够有效缓解Dead ReLU Problem，此时无须刻意调小学习率，能够在收敛速度和运算结果间保持较好的平衡。

- BN层是目前大部分深度学习模型的标配，但前提是你有能力去对其进行优化；

## 4 学习率调度基本概念与手动实现方法

学习率作为模型优化的重要超参数，在此前的学习中，我们已经看到了学习率的调整对模型训练在诸多方面的显著影响。这里我们先简单回顾此前遇到的学习率调整的场景：

- 缓解Dead ReLU Problem

在ReLU激活函数叠加的神经网络中，由于ReLU本身负值归零的特性，可能会出现Dead ReLU Problem，根据Lesson 13.4的实验结果，我们可以通过减小学习率来降低模型落入活性失效陷进的概率。不过降低学习率也代表模型收敛速度更慢；

- 提升模型表现

同时，学习率调整的也将显著影响模型性能。在学习率绝对数值的调整过程中，学习率对模型性能的影响整体呈现U型（回归类是U型，准确率评估指标时即分类时是倒U型）特征，即学习率过大或者过小都不好，学习过大可能会导致模型无法穿越狭窄的通道最终抵达最小值点，而学习率太小则容易在最小值点附近停止收敛，因此在进行模型训练时，我们需要找到一个适中的准确率取值（如Lesson 14.3中的0.005）。

整体来看，如果模型学习率设置太大，虽然前期收敛速度较快，但容易出现收敛过程不稳定、收敛结果不佳、或者神经元活性失效等问题；而如果学习率设置太小，虽然收敛过程将相对平稳，并且能够有效规避神经元活性坏死的问题，但容易出现收敛速度慢、收敛结果不佳等问题。为了深入理解该问题，同时也为了为后续实验储备对比数据，首先，我们可以通过设计实验来观测不同超参数取值对模型收敛速度、收敛结果影响。

### 4.1 学习率对模型训练影响

模型仅lr不同的每批次的训练损失图：

![image-20230812083157978]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_010.png' | relative_url }})

由此我们能够清楚的看到，学习率较大的模型收敛速度较快，但效果却不一定最好。

尽管学习率的调整会对模型训练造成很大的影响，但在进行学习率优化的时候一般不会采用同时训练多组学习率采用不同取值的模型来对比选择最佳学习率的方法。在实际生产生活中，训练一个模型已经是非常耗费时间的了，为了一个超参数同时训练多组模型并不划算。而要通过学习率数值调整来让模型“又好又快”的收敛，就需要采用一种名为学习率调度的优化算法。

### 4.2 学习率调度基本概念与手动实现方法

#### 4.2.1 模型调度基本概念

既然我们无法通过训练多组模型来寻找最优学习率，那么一个最基本的想法，就是让学习率伴随模型训练过程动态调整。并且，根据上述实验结果我们也不难发现，一个比较好的学习率动态调整策略是，先让学习率取得较大数值，从而能够让模型在最开始能够以较快的速度收敛；然后在经过一段时间迭代之后，将学习率调小，从而能够让收敛过程穿过损失函数的“隘口”，抵达更小的解。

这样的一个学习率调整的策略，也被称为学习率调度（Learning rate scheduler）策略。

此处学习率的调整和一般的超参数调整不太一样。一般来说，模型的超参数调整是需要找出一个有助于模型提升效果的确定性的数值，比如模型层数、激活函数、归一化方法选取等，而一旦超参数数值确定，无论是训练过程还是测试过程都将以该参数的取值为准。但学习率参数本质上并不是一个影响模型结构的参数，而是辅助训练过程的参数，更准确的来说，是辅助模型各线性层（目前为止）参数取得最优解的参数。因此，学习率确实是否是确定的值其实并不重要，学习率取值可以随着迭代过程不断变化，只要这样的一个动态变化的过程最终能够让模型“又好又快”的收敛、即让各线性层参数以较快速度收敛至最小值。

##

#### 4.2.3 本地实现方法

而要在 PyTorch 中实现学习率的动态调整，首先我们能够想到的是在fit过程手动调整学习率，例如，第一次fit设置2轮epochs、lr设置为0.03，第二次fit设置2轮epochs、lr设置0.01等等依此类推。我们可以借助Python中的input函数来手动实现学习率伴随迭代epoch次数的动态调整，具体操作方法如下：

- 更新fit函数

```python
def fit_rec(net,
            criterion,
            optimizer,
            train_data,
            test_data,
            epochs = 3,
            cla = False,
            eva = mse_cal):
    """模型训练函数（记录每一次遍历后模型评估指标）

 :param net：待训练的模型
 :param criterion: 损失函数
 :param optimizer：优化算法
 :param train_data：训练数据
 :param test_data: 测试数据
 :param epochs: 遍历数据次数
 :param cla: 是否是分类问题
 :param eva: 模型评估方法
 :return：模型评估结果
 """
    train_l = []
    test_l = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_data:
            if cla True:
                y = y.flatten().long() # 如果是分类问题，需要对y进行整数转化
                yhat = net.forward(X)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.eval()
                train_l.append(eva(train_data, net).detach())
                test_l.append(eva(test_data, net).detach())
                return train_l, test_l
```

- 手动动态调整过程

接下来，我们尝试借助Python中的input函数功能，来执行手动动态调整模型训练过程中的学习率。

所谓手动调整学习率，指的是可以在自定义的训练间隔中灵活调整模型学习率，而要将手动输入的数值作为模型当前阶段训练的参数，就需要使用input函数。Python中的input函数通过提供用户可交互的输入窗口来捕捉用户的即时输入，并将其转化为字符串形式传入当前操作空间中。

```python
 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model = net_class2(act_fun=torch.tanh, in_features=5, BN_model='pre
```

```python
# 创建用于保存记录结果的空列表容器
train_mse = []
test_mse = []

# 创建可以捕捉手动输入数据的模型训练流程
while input("Do you want to continue the iteration? [y/n]") "y": # 询问是否继续迭代
    epochs = int(input("Number of epochs:")) # 下一轮迭代遍历几次数据
    lr = float(input("Update learning rate：")) # 设置下一轮迭代的学习率
    train_l, test_l = fit_rec(net = tanh_model,
                              criterion = nn.MSELoss(),
                              optimizer = optim.SGD(tanh_model.parameters(), lr = lr),
                              train_data = train_loader,
                              test_data = test_loader,
                              epochs = epochs,
                              cla = False,
                              eva = mse_cal)
    train_mse.extend(train_l)
    test_mse.extend(test_l)
```

```python
plt.plot(train_mse, label='train_mse')
plt.legend(loc = 1)
```

![image-20230812085844059]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_011.png' | relative_url }})

同时，我们可以让模型以0.03作为学习率迭代60个epochs，对比模型训练效果

```python
# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model = net_class2(act_fun=torch.tanh, in_features=5, BN_model='pr

# 创建用于保存记录结果的空列表容器
train_msel = []
test_msel = []

# 创建可以捕捉手动输入数据的模型训练流程
while input("Do you want to continue the iteration? [y/n]") "y": # 询问是否继续迭代
    epochs = int(input("Number of epochs:")) # 下一轮迭代遍历几次数据
    lr = float(input("Update learning rate：")) # 设置下一轮迭代的学习率
    train_l, test_l = fit_rec(net = tanh_model,
    criterion = nn.MSELoss(),
    optimizer = optim.SGD(tanh_model.parameters(), lr = lr),
    train_data = train_loader,
    test_data = test_loader,
    epochs = epochs,
    cla = False,
    eva = mse_cal)
    train_msel.extend(train_l)
    test_msel.extend(test_l)
```

```python
plt.plot(train_mse, label='train_mse')
plt.plot(train_msel, label='train_msel')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend(loc = 1)
```

![image-20230812090005523]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_012.png' | relative_url }})

能够看出，手动调整模型学习率的第一个模型在最终模型表现上比第二个模型更好，在同样遍历了60次的情况下，模型一能够收敛至一个更小的MSE。

#### 4.2.4 基于tensorboard实现

#### 4.2.5 常用学习率调度思路

根据上述实验，我们基本能够总结得出，伴随模型遍历数据集次数增加、学习率逐渐降低，能够使得模型整体性能有所提升。但是，上述手动实验过程中学习率调整的方法还是相对来说比较粗糙，我们只是每隔30个epochs将学习率调小一次，很多结果也是通过尝试最后得出。在实际生产工作中，我们需要使用理论更加严谨的学习率调度方法。
在长期的实践经验总结基础上，目前来看，从实现思路上划分，比较通用的学习率调度的策略分为以下五类：

- 幂调度：

随着迭代次数增加，学习率呈现幂律变化，例如，lr为初始（第一轮）学习率，第二轮迭代时学习率调整为lr/2，第三轮迭代时学习率调整为lr/3等；（实践过程具体数值有所差异）

- 指数调度：

随着迭代次数增加，学习率呈现指数变化，例如，lr为初始（第一轮）学习率，第二轮迭代时学习率调整为lr/10^1^，第三轮迭代时学习率调整为lr/10^2^等；（实践过程具体数值有所差异）

- 分段恒定调度：

即每隔几轮迭代调整一次学习率，例如1-10轮学习率lr为，10-20轮时学习率为lr/100等，不难发现此前我们所做的实验就是一种特殊的分段恒定调度策略；

- 性能调度：

即每隔一段时间观察误差变化情况，如果**误差基本不变，则降低学习率继续迭代**；

- 周期调度：

和前面几种学习率调度策略一味将学习率递减有所不同，**周期调度允许学习率在一个周期内进行先递增后递减的变化**；

### 4.3 学习率调度在PyTorch中的实现方法

学习率调度作为模型优化的重要方法，也集成在了 PyTorch 的`optim`模块中。我们可以通过下述代码将学习率调度模块进行导入

```python
from torch.optim import lr_scheduler
```

#### 4.3.1 优化器与状态字典（state_dict）

模型优化器是求解损失函数的函数，其中包含了模型训练的诸多关键信息，包括模型参数、模型学习率等，同时在进行模型训练时，我们也是通过优化器调整模型参数、归零模型梯度。而在学习率调度过程中，由于我们需要动态调整学习率，而学习率又是通过传入优化器进而影响模型训练的，因此在利用 PyTorch 进行学习率调度的时候，核心需要考虑的问题是如何**让优化器内的学习率随着迭代次数增加而不断变化**。

```python
# 创建优化器
optimizer = torch.optim.SGD(tanh_model1.parameters(), lr=0.01)
```

- 在优化器创建完成之后，我们可以使用`.state_dict()`方法查看优化器状态。

```python
optimizer.state_dict()
'''{'state': {},
 'param_groups': [{'lr': 0.01,
 'momentum': 0,
 'dampening': 0,
 'weight_decay': 0,
 'nesterov': False,
 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]}'''
optimizer.state_dict()['param_groups'][0]['lr']
'''0,01'''
```

该字典包含两个元素，第一个是**优化器状态**（state），第二个是优化器相关**参数簇**（param_groups），其中，目前为止核心需要关注的是参数簇中的lr对象，该对象代表着下一次模型训练的时候所带入的学习率。**params**表示**训练参数个数（其中一个矩阵（张量）算作一个参数)**

```python
# 验证带训练参数个数，
len(list(tanh_model1.parameters()))
'''10'''
```

#### 4.1.2 模型(、优化器)本地保存与读取方法
同时，借助state_dict()方法，我们可以实现模型或优化器的本地保存于读取。此处以模型为例，优化器的本地保存相关操作类似。

对于模型而言，其实也有`state_dict()`方法。通过该方法的调用，可以查看模型全部参数信息。

```python
tanh_model1.state_dict()
'''OrderedDict([('linear1.weight',
 tensor([[ 0.2365, -0.1118, -0.3801, 0.0275, 0.4168],
 [-0.1995, -0.1456, 0.3497, -0.0622, -0.1708],
 [-0.0901, 0.0164, -0.3643, -0.1278, 0.4336],
 [-0.0959, 0.4073, -0.1746, -0.1799, -0.1333]])),
 ('linear1.bias', tensor([-0.3999, -0.2694, 0.2703, -0.3355])),
 ('normalize1.weight', tensor([1., 1., 1., 1.])),
 ('normalize1.bias', tensor([0., 0., 0., 0.])),
 ('normalize1.running_mean', tensor([0., 0., 0., 0.])),
 ('normalize1.running_var', tensor([1., 1., 1., 1.])),
 ('normalize1.num_batches_tracked', tensor(0)),
 ('linear2.weight',
 tensor([[ 0.1708, 0.4704, -0.0635, 0.2187],
 [ 0.2336, -0.3569, -0.1928, -0.1566],
 [ 0.4825, -0.4463, 0.3027, 0.4696],
 [ 0.3953, 0.2131, 0.2226, -0.0267]])),
 …………
 ………………
'''
```

值得注意的是，**模型的训练和保存，本质上都是针对模型的参数**。而模型的`state_dict()`则**包含了模型当前全部的参数信息**。因此，保存了模型的`state_dict()`就**相当于**是**保存了模型。**

1) 首先，我们可以将该存有模型全部参数信息的字典对象赋给某个变量。

```python
# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model1 = net_class2(act_fun= torch.tanh, in_features=5, BN_model='pr)
# 将该存有模型全部参数信息的字典对象赋给某个变量
t1 = tanh_model1.state_dict()
```

2) 也可以通过torch**.save()**来将该参数保存至本地。

```python
torch.save(tanh_model1.state_dict(), 'tanh1.pt')
```

对于torch.save函数来说，**第一个参数是需要保存的模型参数，而第二个参数则是保存到本地的文件名**。一般来说可以令其后缀为`.pt`或`.pth`。而当我们需要读取保存的参数结果时，则可以直接使用**load_state_dict(fileName)**方法。

当需要还原tanh_model1中原始参数，我们只能考虑通过使用load_state_dict方法，将本次保存的原模型参数替换当前的tanh_model1中参数。首先需要totch**.load(文件名)**加载，再去覆盖：

```python
tanh_model1.load_state_dict(torch.load('tanh1.pt'))
```

**model.state_dict()** -->**torch.save**(model.state_dict() , '文件名') -->tanh_model1**.load_state_dict**(torch**.load**('文件名'))

至此，我们就完成了模型训练与保存的基本过程。当然，除了模型可以按照上述方法保存外，**优化器也可以类似进行本地存储**。

#### 4.3.3 LambdaLR基本使用方法

接下来，我们通过调用**optim模块中lr_scheduler**相关函数，来**实现优化器中学习率的动态调整**。

**让优化器动态调整学习率的类，也被我们称为学习率调度器类**，该**类实例化的对象**也被称为学习率调度器。在所有的学习率调度器中，LambdaLR类是实现学习率调度**最简单灵活**、同时也是最通用的一种方法。

lr_scheduler.LambdaLR(optim优化器，lambda匿名函数)

要使用LambdaLR来完成学习率调度，首先需要准备一个lambda匿名函数，例如：

```python
lr_lambda = lambda epoch: 0.5 ** epoch
```

该函数需要输入一个参数，一般来说我们会将该参数视作模型迭代次数。当然上述匿名函数是个非常简单的匿名函数，输出结果就是0.5的epoch次方。

> 此处需要注意，一般来说epoch取值从0开始，并且用于学习率调度的匿名函数参数取值为0时，输出结果不能为0。

- 创建学习率调度器

```python
# 创建学习率调度器
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

LambdaLR学习率调度器的创建必须要输入一个lambda函数和与之关联的优化器。一旦优化器创建完成，我们即可继续观察优化器optimizer的状态。

```python
optimizer.state_dict()
'''{'state': {},
 'param_groups': [{'lr': 0.05,
 'momentum': 0,
 'dampening': 0,
 'weight_decay': 0,
 'nesterov': False,
 'initial_lr': 0.05,
 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]}'''
```

此时优化器的参数簇中多了'initial_lr'元素。该元素代表初始学习率，也就是我们在实例化优化器时输入的学习率。而优化器中的lr，则仍然表示下一次迭代时的学习率。

对于LambdaLR学习调度来说，优化器中的lr伴随模型迭代相应调整的方法如下：
$$
lr=lr\_lambda(epoch)*initial\_lr
$$
并且，第一次实例化LambdaLR时epoch取值为0时，因此此时优化器的lr计算结果如下：
$$
lr_0=0.5^0*0.05=0.05
$$
而在后续计算过程中，每当我们调用一次**scheduler.step()，epoch数值就会+1**。我们可以进行下述实验，即当一轮训练完成时，我们可通过scheduler.step()来更新下一轮迭代时的学习率。

```python
for X, y in train_loader:
 yhat = tanh_model1.forward(X)
 loss = criterion(yhat, y)
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
scheduler.step()
```

> 1.之所以将**学习率调度器**放在模型小批量梯度下降循环的外侧，也是因为一般来说遍历一次完整训练集（一个epoch）才会对学习率进行一次更新，而不是每次计算完一个小批数据就对模型学习率进行更新。

> 2.此处需要注意，PyTorch 中要**求先进行优化器的step，再进行学习率调度的step**，此处需要注意**先后顺序**。另外，上述过程之所以提前将优化器内保存的模型参数清零，也是为了防止上述实验过程最终导致模型参数被修改（梯度为0时模型无法修改参数）。

### 4.4 LambdaLR学习率调度实验

#### 4.4.1 前期准备与匿名函数定义

在实验开始前，我们需要将之前定义的fit_rec函数再次进行改写，新函数需要包含学习率调度相关方法。

```python
def fit_rec_sc(net,
               criterion,
               optimizer,
               train_data,
               test_data,
               scheduler,
               epochs = 3,
               cla = False,
               eva = mse_cal):
    """加入学习率调度后的模型训练函数（记录每一次遍历后模型评估指标）

 :param net：待训练的模型
 :param criterion: 损失函数
 :param optimizer：优化算法
 :param train_data：训练数据
 :param test_data: 测试数据
 :param scheduler: 学习率调度器
 :param epochs: 遍历数据次数
 :param cla: 是否是分类问题
 :param eva: 模型评估方法
 :return：模型评估结果
 """
    train_l = []
    test_l = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_data:
            if cla True:
                y = y.flatten().long() # 如果是分类问题，需要对y进行整数转化
                yhat = net.forward(X)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                net.eval()
                train_l.append(eva(train_data, net).detach())
                test_l.append(eva(test_data, net).detach())
                return train_l, test_l
```

定义一个衰减速度更加缓慢的学习率调度器:

```python
lr_lambda = lambda epoch: 0.95 ** epoch
```

```python
# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model1 = net_class2(act_fun=torch.tanh, in_features=5, BN_model='pre')
```

```python
# 创建优化器
optimizer = torch.optim.SGD(tanh_model1.parameters(), lr=0.05)
```

```python
# 创建学习率调度器
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### 4.4.2 模型训练与结果比较

```python
# 进行模型训练
train_l, test_l = fit_rec_sc(net = tanh_model1,
criterion = nn.MSELoss(),
optimizer = optimizer,
train_data = train_loader,
test_data = test_loader,
scheduler = scheduler,
epochs = 60,
cla = False,
eva = mse_cal)
```

```python
plt.plot(train_l, label='train_mse')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend(loc = 1)
```

![image-20230812152311169]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_013.png' | relative_url }})

- 对比恒定学习率为0.03时模型训练结果

```python
# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
tanh_model1 = net_class2(act_fun= torch.tanh, in_features=5, BN_model='pre')

# 进行模型训练
train_l3, test_l3 = fit_rec_sc(net = tanh_model1,
criterion = nn.MSELoss(),
optimizer = optimizer,
train_data = train_loader,
test_data = test_loader,
epochs = 60,
cla = False,
eva = mse_cal)

```

```python
plt.plot(train_l, label='train_l')
plt.plot(train_l3, label='train_l3')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend(loc = 1)
```

![image-20230812152751916]({{ '/assets/learning/11_normalization_and_batch_normalization/figure_014.png' | relative_url }})

在训练了60轮之后，LambdaLR模型最终学习率在0.02附近，相比上述0.01学习率模型而言学习率更小。但从上述的实验中我们发现，恒定学习率时从恒定0.03到恒定0.01的过程，模型准确率已经发生了明显的下降，但在如果是采用动态调整学习率的策略，则可以在一个最终更小的学习率取值的情况下取得一个更好的模型结果。

```python
lr_lambda(60) * 0.05
'''0.002303489949347597'''
```

这其实说明损失函数在超平面空间的图像比一般的想象要复杂的多，很多时候并不是越靠近全域最小值点附近的通道就越窄，会导致迭代过程落入局部最小值陷阱的学习率大小取值也只是绝对概念。正是由于损失函数的复杂性，才导致很多时候我们认为神经网络的内部训练是个“黑箱”，才进一步导致神经网络的模型训练往往以模型结果为最终依据，这也是神经网络优化算法会诞生诸多基本原理层面比较扎实，但却找不到具体能够证明优化效果的理论依据的方法。

不过，针对此类方法，和此前介绍的 Batch Normalization 一样，尽管理论层面无法具体整体优化效果，但对于使用者来说仍然需要在了解其底层原理基础上积累使用经验或者调参经验。因此在后续的课程中，我们将在继续介绍其他学习率优化方法的同时，通过大量的实践来快速积累使用经验，并且在更多事实的基础上找到解释和理解的角度。
