---
layout: article
title: "多层神经网络"
key: learning-04-multilayer-neural-network
permalink: /learning/04-multilayer-neural-network.html
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

从单层到多层是神经网络发展史上的重大变化，层的增加彻底将神经网络的性能提升到了另一个高度， 正确理解层的意义对于我们自主构建神经网络有很重要的作用，学会利用层是避免浪费计算资源以及提 升神经网络效果的关键。在本节的最后，我们还将继承nn.Module类实现一个完整的深层神经网络的前向传播的过程 。

## 1.1 异或门问题

“与门”（andgate）：与门是一组只有两个特征（x1，x2）的分 类数据，当两个特征下的取值都为1时，分类标签为1，其他时候分类标签为0。

在机器学习中，存在一种对数据进行分类的简单方式：就是在两类数据中间绘制一条直线，并规定直线 一侧的点属于一类标签，下方另一侧的点属于另一类标签。而在这个由 与 构成的空间中，二维平面 上的任意一条线可以被表示为：
$$
x_1 = ax_2+b
$$
表达式变换一下：

$$
\begin{aligned}
&0=b-x_1+ax_2 \\
&0=[1,x_1,x_2]*\begin{bmatrix}b\\-1\\a\end{bmatrix} \\
&0=\boldsymbol{X}\boldsymbol{w}
\end{aligned}
$$
其中[b, -1, a]就是我们的权重向量 $ \omega$（默认列向量）， $X$就是我们的特征张量， $b$是我们的截距。在一组 数据下，给定固定的 $X$和 $b$，这个式子就可以是一条固定直线，在 $ \omega$和 $b$不确定的状况下，这个表达式 $X\omega = 0$ 就可以代表平面上的任意一条直线。在直线上方的点为1类（红色），在直线下方的点为0类 （紫色），其颜色标注与刚才绘制的与门数据中的数据点一致。

![1691132434785]({{ '/assets/learning/04_multilayer_neural_network/figure_001.png' | relative_url }})

如果在$ \omega$和 $b$固定时，给定一个唯一的 $X$的取值，这个表达式就可以表示一个固定的点。如果任取一个紫色的点$X_p$ 就可以被表示为：
$$
X_p \omega = p
$$
由于紫色的点所代表的标签y是0，所以我们规定，p<=0。

同样的，对于任意一个红色的点$X_r$而言，我 们可以将它表示为 :
$$
X_r \omega = r
$$
由于红色点所表示的标签y是1，所以我们规定，r>0。由此，如果我们有新的测试数据$X_t$ ，则的 $X_t$ 标签就可以根据以下式子来判定：
$$
y=\left\{\begin{matrix}1&&(X_t\omega>0)\\0&&(X_t\omega\leq0)\end{matrix}\right.
$$
在我们只有两个特征的时候，实际上这个式子就是：

$$
y=\left\{\begin{matrix}1&&if w_1x_1+w_2x_2+b>0\\0&&ifw_1x_1+w_2x_2+b\leq0\end{matrix}\right.
$$
而$\omega_1x_1+\omega_2x_2+b$就是我们在神经网络中表示 $z$的式子，所以这个式子实际上就是：

用线划分点的数学表达和我们之前学习的阶跃函数的表示方法一模一样！

```python
import torch
X = torch.tensor([[1,0,0],[1,1,0],[1,0,1],[1,1,1]], dtype = torch.float32)
andgate = torch.tensor([0,0,0,1], dtype = torch.float32)
def AND(X):
    w = torch.tensor([-0.2,0.15, 0.15], dtype = torch.float32)
    zhat = torch.mv(X,w)
    andhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return andhat
andhat = AND(X)
andgate
```

实际上，**阶跃函数的本质也就是用直线划分点的过程**，而这条具有分类功能直线被我们称为“**决策边界**”。在机器学习中，任意分类算法都可以绘制自己的决策边界，且依赖决策边界来进行分类。（逻辑回归自然也有，不过因为sigmoid函数的决策边界绘制起来没有阶跃函数那么直接，因此我们没有选择逻 辑回归的决策边界来进行绘制）。

不难看出，既然**决策边界的方程就是$z$的表达式**，那在相同的数据 $X$下 （即在相同的数据空间中），**决策边界函数具体在哪里就是由我们定义的$\omega$ 决定的**。

对于任意的数据，我们只需要找到适合的$\omega$ 和 $ b$就能够确认相应的决策边界，也就可以自由进行分类了。

阶跃函数作为线性结果$z$之后的函数，在以下典型数据上使用决策边界进行分类 ：

**或门（OR GATE）**：特征一或特征二为1的时候标签就为1的数据

```python
#或门的函数（基于阶跃函数）
def OR(X):
    w = torch.tensor([-0.08,0.15,0.15], dtype = torch.float32) #在这里我修改了b的数
    值
    zhat = torch.mv(X,w)
    yhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return yhat
```

**非与门（NAND GATE）**：特征一和特征二都是1的时候标签就为0，其他时候标签则为1的数据。

```python
def NAND(X):
    w = torch.tensor([0.23,-0.15,-0.15], dtype = torch.float32) #和与门、或门都不同
    的权重
    zhat = torch.mv(X,w)
    yhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return yhat
```

以上三类可以为用**单层神经网络**实现，找粗**直线**决策边界函数。

现在，来看看下面这一组数据：

异或门（XOR） ：

异或门（XOR）的真值表如下：

| $x_0$ | $x_1$ | $x_2$ | $y_{\text{xor}}$ |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 0 |

如何把直线的决策边界变成曲线呢？答案是将单层神经网络变成多层。来看下面的 网格结构：

![1691134876601]({{ '/assets/learning/04_multilayer_neural_network/figure_007.png' | relative_url }})

这是一个多层神经网络，除了输入层和输出层，还多了一层“中间层” 。

```python
def XOR(X):
    #输入值：
    input_1 = X
    #中间层：
    sigma_nand = NAND(input_1)
    sigma_or = OR(input_1)
    x0 = torch.tensor([[1],[1],[1],[1]],dtype=torch.float32)
    #输出层：
    input_2 = torch.cat((x0,sigma_nand.view(4,1),sigma_or.view(4,1)),dim=1)
    y_and = AND(input_2)
    #print("NANE:",y_nand)
    #print("OR:",y_or)
    return y_and

'''tensor([0., 1., 1., 0.])'''
```

最终输出的结果和异或门要求的结果是一致的。可见，拥有更多的”层“帮助我们解决了单层 神经网络无法处理的非线性问题。叠加了多层的神经网络也被称为”多层神经网络“。多层神经网络是神经网络在深度学习中的基本形态，接下来，我们就来认识多层神经网络。

以上都为Sign符号函数。

## 1.2 黑箱：深层神经网络的不可解释性

![1691134876601]({{ '/assets/learning/04_multilayer_neural_network/figure_007.png' | relative_url }})

从结构上来看，多层神经网络比单层神经网络多出了“中间层”。中间层常常被称为**隐藏层**（hidden layer），理论上来说可以有无限层，所以在图像表示中经常被省略。层数越多，神经网络的模型复杂度 越高，一般也认为更深的神经网络可以解决更加复杂的问题。在学习中，通常我们最多只会设置3~5个 隐藏层，但在实际工业场景中会更多。还记得这张图吗？当数据量够大时，现代神经网络层数越深，效果越好。

![1691135913498]({{ '/assets/learning/04_multilayer_neural_network/figure_008.png' | relative_url }})

在一个神经网络中，更靠近输入层的层级相对于其他层级叫做"上层"，更靠近输出层的则相对于其他层级叫做"下层"。若从**输入层开始从左向右编号**，则**输入层为第0层**，输出层为最后一层。除了输入层以 外，每个神经元中都存在着对数据进行处理的数个函数。在我们的例子异或门（XOR）中，隐藏层中的 函数是NAND函数和OR函数（也就是线性回归的加和函数+阶跃函数），输出层上的函数是AND函数。 对于所有神经元和所有层而言，**加和函数的部分都是一致的（都得到结果 $z$）**，因此我们需要关注的是**加和之外的那部分函数。在隐藏层中这个函数被称为 激活函数**$h(z)$，符号为 ，在输出层中这个函数只是普通的连接函数$g(z)$，我们定义为是 。我**们的数据被逐层传递，每个下层的神经元都必须处理上层的神经元中的激活函数$h(z)$处理完毕的数据**，整个流程本质是一个嵌套计算结果的过程 。

![1691216432375]({{ '/assets/learning/04_multilayer_neural_network/figure_003.png' | relative_url }})

在神经网络中，任意层上都有至少一个神经元，最上面的是**常量神经元**，连接常量神经元的箭头上的参数是截距 $b$，剩余的是**特征神经元**，连接这些神经元的箭头上的参数都是权重 $\omega$。**神经元是从上至下进行编号**，需要注意的是，**常量神经元与特征神经元是分别编号的。和从0开始编号的层数不同，神经元是从 1开始编号的**。在异或门的例子中，含有1的偏差神经元是1号偏差神经元，含有特征 的神经元则是1号 特征神经元 。

**层数从0开始编号；神经元从1开始编号，且常量神经元与特征神经元是分别编号的**

除了神经元和网络层，权重、偏差、神经元上的取值也存在编号。这些编号规律分别如下：

**上标为层的编号，下标为层间神经元的编号**。$b$为常量神经元；$w$为特征神经元；$z$为加和函数∑的结果，$\sigma$是对z经过激活函数h(z)后的结果。**都从1开始编号**

![1691216796383]({{ '/assets/learning/04_multilayer_neural_network/figure_004.png' | relative_url }})

这些编号实在很复杂，将编号改写为如下情况：

![1691217657884]({{ '/assets/learning/04_multilayer_neural_network/figure_005.png' | relative_url }})

有了这些编号说明，我们就可以用数学公式来表示从输入层传入到第一层隐藏层的信号了。以上一节中 说明的XOR异或门为例子，对于仅有两个特征的单一样本而言，在第一层的第一个特征神经元中获得加 和结果的式子可以表示为：
$$
z_1^1=b_{1}^{1}+x_1w_{11}^{1}+x_2w_{21}^{1}
$$
而隐藏层中被 处理的公式可以写作：
$$
\begin{aligned}
\sigma_{1}^{1}& =h(z_1^1) \\
&=h(b_{\to1}^{\to1}+x_1w_{1\to1}^{0\to1}+x_2w_{2\to1}^{0\to1})
\end{aligned}
$$
现在，我们用矩阵来表示数据从输入层传入到第一层，并在第一层的神经元中被处理成 的情况：
$$
\begin{aligned}
Z^{1}& =\mathbf{W}^1\cdot\mathbf{X}+\mathbf{B}^1 \\
\left[\begin{smallmatrix}z_1^1\\z_2^1\end{smallmatrix}\right]& =\left[{\begin{array}{c}w_{1\to1}^{0\to1}w_{1\to2}^{0\to1}\\w_{2\to1}^{0\to1}w_{2\to2}^{0\to1}\end{array}}\right]*\left[{\begin{array}{c}x_{1}\\x_{2}\end{array}}\right]+\left[{\begin{array}{c}b_{\to1}^{\to1}\\b_{\to2}^{\to1}\end{array}}\right] \\
\text{矩阵结构表示为:}(2,1)& =(2,2)*(2,1)+(2,1) \\
&=\begin{bmatrix}x_1w_{1\to1}^{0\to1}+x_2w_{1\to2}^{0\to1}\\x_1w_{2\to1}^{0\to1}+x_2w_{2\to2}^{0\to1}\end{bmatrix}+\begin{bmatrix}b_{\to1}^{\to1}\\b_{\to2}^{\to1}\end{bmatrix} \\
&=\left[\left.\begin{aligned}x_1w_{1\to1}^{0\to1}+x_2w_{1\to2}^{0\to1}+b_{\to1}^{\to1}\\x_1w_{2\to1}^{0\to1}+x_2w_{2\to2}^{0\to1}+b_{\to2}^{\to1}\end{aligned}\right]\right. \\
\left[\begin{smallmatrix}\sigma_1^1\\\sigma_1^1\\\sigma_2^1\end{smallmatrix}\right]& =\begin{bmatrix}h(x_1w_{1\to1}^{0\to1}+x_2w_{1\to2}^{0\to1}+b_{\to1}^{\to1})\\h(x_1w_{2\to1}^{0\to1}+x_2w_{2\to2}^{0\to1}+b_{\to2}^{\to1})\end{bmatrix}
\end{aligned}
$$
相应的从中间层最下面的神经网络会得到的结果是 $\sigma_1^{2}$（如果是阶跃函数则是直接得到y）。 会作为中间 层的结果继续传入下一层。如果我们继续向下嵌套，则可以得到：
$$
\begin{aligned}
&z_{1}^{2} =b_{\to1}^{\to2}+\sigma_1^1w_{1\to1}^{1\to2}+\sigma_2^1w_{2\to1}^{1\to2} \\
&\sigma_{1}^{2} =g(z_1^2) \\
&\sigma_{1}^{2} =g(b_{\to1}^{\to2}+\sigma_1^1w_{1\to1}^{1\to2}+\sigma_2^1w_{2\to1}^{1\to2})
\end{aligned}
$$
真实数据可能有超过数百甚至数千个特征，所以真实神经网络的复杂度是非常高，计 算非常缓慢的。所以，当神经网络长成如下所示的模样，我们就无法理解中间过程了。我们不知道究竟 有多少个系数，如何相互作用产生了我们的预测结果，因此神经网络的过程是一个“黑箱” 。

![1691219975533]({{ '/assets/learning/04_multilayer_neural_network/figure_006.png' | relative_url }})

在 PyTorch 中实现神经网络的时候，我们一般用不到这些复杂的数学符号，也不需要考虑这些嵌套流 程，毕竟这些计算非常底层。那为什么我们还要学习这些符号呢？有以下几点原因：

1、利用数学的嵌套，我们可以很容易就理解深层神经网络为什么会随着层数的增多、每层上神经元个数 的增多而变得复杂，从而理解“黑箱”究竟是怎样形成的。

2、多层神经网络与单层神经网络在许多关键点上其实有所区别，这种区别使用代数表示形式会更容易显示。比如**，单层神经网络（线性回归、逻辑回归）中直线的表现形式都是 $Xw$**，且**$w$是结构为 (n_features,1)的列向量**，但在多层神经网络中，随着“层”和神经元个数的增加，只有输入层与第一个隐藏层之间是特与$w$的关系，隐藏层与隐藏层、隐藏层与输出层之间都是$ \sigma$与 $w$的关系。并且，即便是在输入层与第一个隐藏层之间，**单个特征所对应的$w$不再是列向量**，而是**结构为**(上层特征神经元个数，下层特征神经元个数)的矩阵，每两层就有一个权重矩阵，表现为 $WX$，（**实际底层是torch.mm(x, W.t()) + b.t())，其中样本x，截距b都为为行向量**，**见08模型优化算法**）

并且，每两层神经元之间，都会存在一个权重矩阵，权重将无法直接追踪到特征x上，这也是多层神经网络无法被解释的一个关键原因。同时，**为了让输出结果 $z$和$ \sigma$ 都保持列向量 的形式（与神经网络的图像匹配）**， $X$的结构也要顺应的变化而变化（上文的推导中我们展现的是单 一样本仅有两个特征的情况，想想看多个样本多个特征会是什么样吧），相乘公式也变化为 。这些 细节在由单层推广到多层时，都会成为新手容易掉入的坑，基础不牢固的情况下，新手很可能在2层推广 到4层，2个特征推广到N个特征，甚至是1个样本推广到多个样本时被卡住。

3、嵌套的数学公式可以帮助我们更好地理解反向传播，以及更好地阅读其他教材。

不过，即便神经网络是一个神秘黑箱，我们依然可以对它进行一系列的探索。基于我们已经熟悉的三层 神经网络XOR的结构，我们来提出问题。

## 1.3 探索多层神经网络：层 vs h(z)

想要发问的隐藏层的作用。在之前的XOR函数中，我们提出”多层神经网络能够描绘出一条曲线 作为决策边界，以此为基础处理单层神经网络无法处理的复杂问题“，这可能让许多人产生了“是增加层 数帮助了神经网络”的错觉。实际上并非如此。

在神经网络的隐藏层中，存在两个关键的元素，一个是加和函数 ∑，另一个是 $h(z)$。除了输入层之外， 任何层的任何神经元上都会有加和的性质，因为神经元有“多进单出”的性质，可以一次性输入多个信号，但是输出只能有一个，因此输入神经元的信息必须以某种方式进行整合，否则神经元就无法将信息传递下去，而最容易的整合方式就是加和 ∑。因此我们可以认为加和是神经元自带的性质，只要增加 更多的层，就会有更多的加和。但是$h(z)$的存在却不是如此，即便隐藏层上没有 （或是一个恒等函数$h(z)=z$），神经网络依然可以从第一层走到最后一层。让我们来试试看，在XOR中，假设隐藏层上没有$h(z)$ 的话， 会发生什么：

```python
#回忆一下XOR数据的真实标签
xorgate = torch.tensor([0,1,1,0],dtype=torch.float32)
def AND(X):
    w = torch.tensor([-0.2,0.15, 0.15], dtype = torch.float32)
    zhat = torch.mv(X,w)
    #下面这一行就是阶跃函数的表达式，注意AND函数是在输出层，所以保留输出层的阶跃函数g(z)
    andhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return andhat
def OR(X):
    w = torch.tensor([-0.08,0.15,0.15], dtype = torch.float32) #在这里我修改了b的数
    值
    zhat = torch.mv(X,w)
    #注释掉阶跃函数，相当于h(z)是恒等函数
    #yhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return zhat
def NAND(X):
    w = torch.tensor([0.23,-0.15,-0.15], dtype = torch.float32)
    zhat = torch.mv(X,w)
    #注释掉阶跃函数，相当于h(z)是恒等函数
    #yhat = torch.tensor([int(x) for x in zhat >= 0],dtype=torch.float32)
    return zhat
def XOR(X):
    #输入值：
    input_1 = X
    #中间层：
    sigma_nand = NAND(input_1)
    sigma_or = OR(input_1)
    x0 = torch.tensor([[1],[1],[1],[1]],dtype=torch.float32)
    #输出层：
    input_2 = torch.cat((x0,sigma_nand.view(4,1),sigma_or.view(4,1)),dim=1)
    y_and = AND(input_2)
    #print("NANE:",y_nand)
    #print("OR:",y_or)
    return y_and
XOR(X)
'''tensor([0., 0., 0., 0.])'''
```

很明显，此时XOR函数的预测结果与真实的xorgate不一致。当隐藏层的 是恒等函数或不存在时，叠 加层并不能够解决XOR这样的非线性问题。从数学上来看，这也非常容易理解。

从输入层到第1层：
$$
\begin{aligned}
\text{从输入层到第 }1\text{层}: \\
z^{1}& =\mathbf{W}^1\cdot\mathbf{X}+\mathbf{B}^1 \\
\Sigma^{1}& =h(\mathbf{Z}^1) \\
\because h(z)\text{为恒等函数} \\
\therefore\Sigma^{1}=\left[\begin{array}{c}{\sigma_{1}^{1}}\\{\sigma_{2}^{1}}\\\end{array}\right]& =\left[\begin{array}{r}z_1^1\\z_2^1\end{array}\right]
\end{aligned}
$$
从第1层到输出层：
$$
\begin{aligned}
z^2& =\mathbf{W}^2\cdot\mathbf{\Sigma}+\mathbf{B}^2 \\
\left[\begin{smallmatrix}2\\z_1\end{smallmatrix}\right]& =\left[\begin{array}{c}w_{1\to1}^{1\to2}w_{1\to2}^{1\to2}\end{array}\right]*\left[\begin{array}{c}\sigma_1^1\\\sigma_2^1\end{array}\right]+\left[\begin{array}{c}b_{\to1}^{\to2}\end{array}\right] \\
\text{由于公式太长,}& \text{不再给}\boldsymbol{w}\text{写箭头角标,而是直接写普通数字角标}: \\
\left[\begin{smallmatrix}z^2\\1\end{smallmatrix}\right]& =\sigma_1^1w_{11}^2+\sigma_2^1w_{12}^2+b_1^2 \\
&=z_1^1w_{11}^2+z_2^1w_{12}^2+b_1^2 \\
&=(x_1w_{11}^1+x_2w_{12}^1+b_1^1)w_{11}^2+(x_1w_{21}^1+x_2w_{22}^1+b_2^1)w_{12}^2+b_1^2 \\
&=x_1(w_{11}^1w_{11}^2+w_{21}^1w_{12}^2)+x_2(w_{12}^1w_{11}^2+w_{22}^1w_{12}^2)+b_{1}^1w_{11}^2+b_{2}^1w_{12}^2+b_{1}^2 \\
&=x_1W_1+x_2W_2+B\text{,其中}W_1,W_2\text{和}B\text{都是常数}
\end{aligned}
$$
不难发现，最终从输出层输出的结果和第一层的输出结果$x_{1}w_{11}^{1}+x_{2}w_{12}^{1}+b_{1}^{1}$是类似的，**只不过是乘以特征$x_1,x_2$的具体数值不同**。在**没有$h(z)$**时，在层中流动的数据被做了**仿射变换**（affine transformation），**仿射变换后得到的依然是一个线性方程，而这样的方程不能解决非线性问题**。可见，“层”本身不是神经网络解决非线性问题的关键，层上的$h(z)$ 才是。从上面的例子和数学公式中可以看出，如果 $h(z)$是线性函数，或不存在，那增加再多的层也没有用。

那是不是任意非线性函数作为 都可以解决问题呢？让我们来试试看，在XOR例子中如果不使用阶跃 函数，而使用sigmoid函数作为 ，会发生什么。

如果将换成sigmoid函数，XOR结构的神经网络同样会失效！可见，**即便是使用了$h(z)$ ，也不一定能够解决曲线分类的问题。在不适合的非线性函数加持下，神经网络的层数再多也无法起效**。所以，$h(z)$ 是真正能够让神经网络算法“活起来”的关键，没有搭配合适 的神经网络结构是无用 的，而 $h(z)$正是神经网络中最关键的概念之一激活函数（activation function）。

## 1.4 激活函数

激活函数：在人工神经网络的神经元上，根据一组输入定义该神经元的输出结果的函数，就是激活函数。**激活函数一般都是非线性函数**，它**出现在神经网络中除了输入层以外的每层的每个神经元上**。

机器学习中常用的激活函数只有恒等函数（identity function），阶跃函数（sign），sigmoid 函数，ReLU，tanh，softmax这六种，其中Softmax与恒等函数几乎不会出现在隐藏层上，Sign、Tanh 几乎不会出现在输出层上，ReLU与Sigmoid则是两种层都会出现，并且应用广泛。在这里，我们将总结性声明一下输出层的g(z)与隐藏层 的h(z)之间的区别，以帮助大家获得更深的理解 ：

1. 虽然都是激活函数，但隐藏层和输出层上的激活函数作用是完全不一样的。**输出层的激活函数g(z)是为了让神经网络能够输出不同类型的标签而存在的**。**其中恒等函数用于回归，sigmoid函数用于 二分类，softmax用于多分类**。换句说， **仅仅与输出结果的表现形式有关**，**与神经网络的效果无关，也因此它可以使用线性的**恒等函数。

但隐藏层的激活函数就不同了，如我们之前尝试的 XOR，隐藏层上的激活函数的选择会影响神经网络的效果，**而线性的是会让深层神经网络的结构失效**的。隐藏层的激活函数必须是非线性的函数。

2. 在同一个神经网络中，g(z)与h(z)可以是不同的，并且在大多数运行回归和多分类的神经网络时， 他们也的确是不同的。每层上的 可以是不同的，但是同一层上的激活函数必须一致。

若没有特别说明，当我们提到“激活函数”时，特指隐藏层上的激活函数 。 当需要表达输出层上的激活函数时，我们需称其为“输出层激活函数”（out_activation）。

## 1.5 正向传播实现

学到这里，我们已经学完了一个普通深度神经网络全部的基本元素——用来构筑神经网络的结构的层与 激活函数，输入神经网络的数据（特征、权重、截距），并且我们了解从左向右的过程是神经网络的正 向传播（也叫做前向传播，或者向前传播） 。

现在可以使用封装好的torch.nn模块来实现一个完整、多层的神经网 络的正向传播。

假设我们有500条数据，20个特征，标签为3分类。我们现在要实现一个三层神经网络，这个神经网络的 架构如下：第一层有13个神经元，第二层有8个神经元，第三层是输出层。其中，第一层的激活函数是 relu，第二层是sigmoid。我们要如何实现它呢？

最后输出结果一定是：样本数 * 输出层神经元个数(类别)

来看代码(**仔细看到注释**)：

创建的神经网络类必须继承nn.Module，并在_ _ _init_ _中使用了nn.module中的层，并且必须在_ _ _init_ _中显示调用super().__init__()，否侧这个神经网络类不可实例化

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

#确定数据
torch.random.manual_seed(420)
# 样本特征
X = torch.rand((500,20), dtype=torch.float32)
# 分类标签
y = torch.randint(low=0,high=3,size=(500,1), dtype=torch.float32)

#继承nn.Modules类来定义神经网路的架构、完成正向传播
class Model(nn.Module):
    #函数1： __init__去定义神经网络结构
    def __init__(self,in_features=10, out_features=2):
        """
        in_features: 输入该神经网络的特征数目（输入层上的神经元的数目）
        out_features：神经网络的输出的数目（输出层上的神经元的数目）
        神经网络的输入与输出与隐藏层无关，所以可以直接确定一个默认值。
        """
        #重写了__init__，则不会隐式调用父类的__init__，因此需要显示调用
        super(Model,self).__init__()
        # 添加实例属性，实例化隐藏层中的各个层
        #输入层不用写
        #隐藏层第一层
        self.linear1 = nn.Linear(in_features,13,bias=True)
        #隐藏层第二层
        self.linear2 = nn.Linear(13,8,bias=True)
        #输出层
        self.output = nn.Linear(8,out_features,bias=True)

    #函数2：forward神经网络的向前传播过程的函数
    def forward(self,x):
        # 运行一个个被实例化后的层，并且添加激活函数
        # 第一层加和 z1
        z1 = self.linear1(x)
        # 第一层的激活函数是relu
        sigma1 = torch.relu(z1)
        # 第一层加和 z2
        z2 = self.linear2(sigma1)
        # 第二层的激活函数是sigmoid
        sigma2 = torch.sigmoid(z2)
        # 第一层加和 z3
        z3 = self.output(sigma2)
        # 第三层的激活函数是softmax
        sigma3 = F.softmax(z3,dim=1)
        return sigma3

input_ = X.shape[1] #特征的数目
output_ = len(y.unique()) #分类的数目
#实例化神经网络
# 定义人为随机数种子，控制随机性
torch.random.manual_seed(420)
net = Model(in_features=input_, out_features = output_)
net.forward(X) #向前传播
# 一下的net(X)会直接执行__init__函数后的所有函数。（实例化对象时已经调用了__init__），因此与net.forward(X)一样，因为只有一个forwar()函数
# net(X)

net.forward(X).shape #softamx最后输出的概率
# 最后输出结果一定是：样本数*输出层神经元个数
'''torch.Size([500, 3])'''
```

```python
# 计算是WX，第0层输入层为20个特征，即有20个特征神经元，
# 那由输入层到第一层的有20个权重，且要输入到第1层的13个神经元中（则可知道有13个b，torch.Size([13])），则有13次w赋值
# 即由0层到第1层的W为torch.Size([13, 20])。
# 计算是WX，则X(500,20) 会自动转置为(20,500)
net.linear1.weight.shape
'''torch.Size([13, 20])'''
```

nn.Module类的定义代码有一千多行，其_ _init__函数中有许多复杂的内容需要被继承，因此super函数 一定不能漏下。

从nn.Module中继承了这些有用的方法：

net.apply() #对__init__中的所有对象（全部层）都执行同样的操作，批量操作

```python
#属性的继承
net.training #是否用于训练

#方法的继承
net.cuda() #将整个网络转移到GPU上来运行
net.cpu() #将整个网络转移到CPU上来运行

net.apply() #对__init__中的所有对象（全部层）都执行同样的操作，批量操作
#比如，令所有线性层的初始权重w都为0
def initial_0(m):
    print(m)
    if type(m) nn.Linear:
    	m.weight.data.fill_(0)
    print(m.weight)

net.apply(initial_0)
net.linear1.weight

```

net.**parameters()** ：一个迭代器，我们可以通过循环的方式查看里面究竟是什么内容：**每层的全部权重和偏差**

```python
#一个特殊的方法
net.parameters() #一个迭代器，我们可以通过循环的方式查看里面究竟是什么内容：每层的全部权重和偏差
for param in net.parameters():
 print(param)
```

```python
# 三分类的神经网络

#1、确定数据
torch.random.manual_seed(420)
# 样本特征
X = torch.rand((500,20), dtype=torch.float32)
# 分类标签
y = torch.randint(low=0,high=3,size=(500,1), dtype=torch.float32)

#继承nn.Modules类来定义神经网路的结构、完成正向传播
class Model(nn.Module):
    #函数1： __init__定义神经网络结构
    def __init__(self,in_features=10, out_features=2):
        # 调用super().__init__()
        super(Model,self).__init__()
        # 添加实例属性，实例化隐藏层中的各个层
        #实例化隐藏层和输出层
        self.linear1 = nn.Linear(in_features,13,bias=True)
        self.linear2 = nn.Linear(13,8,bias=True)
        self.output = nn.Linear(8,out_features,bias=True)

     #函数2：forward用于神经网络的向前传播过程的函数
     def forward(self,x):
         # 运行一个个被实例化后的每一层，加和、添加激活函数
         # 第一层
         z1 = self.linear1(x)
         sigma1 = torch.relu(z1)
         # 第二层
         z2 = self.linear2(sigma1)
         sigma2 = torch.sigmoid(z2)
         # 第三层（输出层）
         z3 = self.output(sigma2)
         sigma3 = F.softmax(z3,dim=1)
         return sigma3

```
