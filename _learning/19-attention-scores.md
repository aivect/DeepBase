---
layout: article
title: "注意力分数"
key: learning-19-attention-scores
permalink: /learning/19-attention-scores.html
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

在上一节中，我们使用高斯核来对查询和键之间的关系建模。我们可以将上一节中的高斯核函数部分视为注意力评分函数，简称评分函数，然后把这个函数的输出结果输入到softmax函数中进行运算。 通过上述步骤，我们将得到与键对应的值的概率分布（即注意力权重）。 最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。

<img src="{{ '/assets/learning/19_attention_scores/figure_001.png' | relative_url }}" alt="image-20230828120309139" style="zoom:80%;" />

```python
import math
import torch
from torch import nn
from d2l import torch as d2l
```

## 1 掩蔽softmax操作

- softmax操作用于输出一个概率分布作为注意力权重。 在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 例如，为了高效处理小批量数据集， **某些文本序列被填充了没有意义的特殊词元。 为了仅将有意义的词元作为值来获取注意力汇聚， 我们可以指定一个有效序列长度（即词元的个数）， 以便在计算softmax时过滤掉超出指定范围的位置。 通过这种方式，我们可以在下面的`masked_softmax`函数中 实现这样的*掩蔽softmax操作*（masked softmax operation）， 其中任何超出有效长度的位置都被掩蔽并置为0。**

```python

x = torch.tensor([[1,2,3],[1,2,4]])
shape = x.shape
shape
# torch.Size([2, 3])
shape[1]
# 3

>>> x = torch.tensor([1, 2, 3])
>>> x.repeat_interleave(2)
tensor([1, 1, 2, 2, 3, 3])
>>> y = torch.tensor([[1, 2], [3, 4]])
>>> torch.repeat_interleave(y, 2)
tensor([1, 1, 2, 2, 3, 3, 4, 4])
>>> torch.repeat_interleave(y, 3, dim=1)
tensor([[1, 1, 1, 2, 2, 2],
 [3, 3, 3, 4, 4, 4]])
>>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
tensor([[1, 2],
 [3, 4],
 [3, 4]])
>>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0, output_size=3)
tensor([[1, 2],
 [3, 4],
 [3, 4]])
```

repeat()

```python

import torch
x = torch.tensor([1,2,3])
#将一维度的x扩展到三维
xx = x.repeat(4,2,1)
/**
扩展步骤如下(倒着执行)：
1 最后一个维度1：此时将[1,2,3]中的数字直接重复1次，得到[1,2,3]，保持没变
2 倒数第二个维度2：先将上一步骤的结果增加一个维度，得到[[1,2,3]]，然后将最外层中括号中的整体重复2次，得到[[1,2,3],[1,2,3]]
3 倒数第三个维度4：先将上一步骤的结果增加一个维度，得到[[[1,2,3],[1,2,3]]]，然后将最外层中括号中的整体重复4次，得到[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
4 三个维度扩展结束，得到结果。
**/

# 将维度1的个数1变为2
# .repeat(2,1,1)就是1维重复一次，二维重复1次，三维重复2次
torch.arange(40).reshape(1,2,40).repeat(2,1,1)
```

d2l.sequence_mask用来生成一个**掩码矩阵**的。掩码矩阵是一个与输入矩阵形状相同的矩阵，其中有效位置的元素值为1，无效位置的元素值为0。最后三维会输出二维，因此需要转换为原shape，即X.reshape(shape)

法1、

```python
import d2l
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
valid_lens = [2, 3, 1]
shape = (3, 3)
X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

print(X)
'''[[1 1 0]
 [1 1 1]
 [1 0 0]]'''
```

法2、

```python
#生成mask序列
def subsequent_mask( size ):
    subsequent_mask = torch.triu( torch.ones( (1, size, size) ) ) 0
    return subsequent_mask
```

```python
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 参数dim=-1表示在输入张量的最后一个维度上进行softmax操作。例如，如果输入张量X的形状为(2, 3, 4)，则dim=-1表示在最后一个维度上进行softmax操作，即在维度为4的轴上进行softmax
    if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
    else:
    shape = X.shape
    if valid_lens.dim() 1:
    valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
    valid_lens = valid_lens.reshape(-1)
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)
```



```python
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))

#tensor([[[[0.5350, 0.4650, 0.0000, 0.0000],
#          [0.6698, 0.3302, 0.0000, 0.0000]],
#         [[0.4257, 0.3151, 0.2592, 0.0000],
#          [0.3343, 0.2299, 0.4359, 0.0000]]]])
```

```
masked_softmax(torch.rand(2, 2, 4),
               torch.tensor([[[1, 3], [2, 4]]]))
               
#tensor([[[[1.0000, 0.0000, 0.0000, 0.0000],
#          [0.5182, 0.2388, 0.2429, 0.0000]],
#
#         [[0.4222, 0.5778, 0.0000, 0.0000],
#          [0.1823, 0.2637, 0.1907, 0.3633]]])
```



## 2 注意力分数：加性注意力、缩放点积注意力

## 2.1 加性注意力

可学参数：

$\mathbf{W}_k \in \mathbb{R}^{h \times k},\ \mathbf{W}_q \in \mathbb{R}^{h \times q},\ \mathbf{v} \in \mathbb{R}^{h}$ $$ a(\mathbf{k}, \mathbf{q}) = \mathbf{v}^T \tanh\left(\mathbf{W}_k \mathbf{k} + \mathbf{W}_q \mathbf{q}\right) $$ 

- 等价于将key和value合并起来后放入到一个隐藏大小为 $h$、输出大小为 1 的单隐藏层MLP

- 一般来说，当查询和键是不同长度的矢量时， 我们可以使用加性注意力作为评分函数。
- 通过参数矩阵 $W_q$ 和 $W_k$，加性注意力可自动调整查询和键的维度，使其在隐藏空间中对齐，从而解决长度不匹配的问题。

这是把经过预测的query放入这样的注意力分数/权重中，这样做反向传播计算梯度，是带着预测来算的，注意力放到预测上，自然更精准。

```python
#@save
    class AdditiveAttention(nn.Module):
        """加性注意力"""
        def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    # valid_lens是多少对key-value需要
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，使用广播机制加和
        # queries的形状：(batch_size，num_querys(查询的个数)，1，num_hidden)
        # key的形状：(batch_size，1，num_keys(“键－值”对的个数)，num_hiddens)
        # 使用广播方式进行求和(batch_size，num_querys，num_keys，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # 降维后scores的形状：(batch_size，num_querys(查询的个数)，num_keys(“键－值”对的个数))
        #不降维则为(batch_size，num_querys(查询的个数)，num_keys(“键－值”对的个数),1)，因为线性层输出为1层
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，num_keys(“键－值”对的个数)，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

加性注意力实现步骤，相比于缩放点积注意力实在得到scores不同：

1）query经过权重矩阵与keys进过权重矩阵加和并经过tanh（需要升维广播才能加和）：features = queries.unsqueeze(2) + keys.unsqueeze(1)
features = torch.tanh(features)放入隐藏层输入为H，输出为1个神经元的隐藏层

2）再进入一个隐藏层scores = w_v(features).squeeze(-1)，得到注意力分数scores

3）self.attention_weights = masked_softmax(scores, valid_lens)得到注意力权重

4）torch.bmm(self.dropout(self.attention_weights), values)：注意力权重与value加和得到新的词向量矩阵。



## 2.2 缩放点积注意力

如果query和key都是同样的长度 $\mathbf{q},\ \mathbf{k}_i \in \mathbb{R}^{d}$，那么可以 
$$
a(\mathbf{q}, \mathbf{k}_i) = \langle \mathbf{q}, \mathbf{k}_i \rangle / \sqrt{d} 
$$
向量化版本  ：

- $\mathbf{Q} \in \mathbb{R}^{n \times d},\ \mathbf{K} \in \mathbb{R}^{m \times d},\ \mathbf{V} \in \mathbb{R}^{m \times v}$ 
-  - 注意力分数：$a(\mathbf{Q}, \mathbf{K}) = \mathbf{Q}\mathbf{K}^T / \sqrt{d} \in \mathbb{R}^{n \times m}$ 
-  - 注意力池化：$f = \mathrm{softmax}\big(a(\mathbf{Q}, \mathbf{K})\big)\mathbf{V} \in \mathbb{R}^{n \times v}$



使用点积可以得到计算效率更高的评分函数， 但是**点积操作要求查询和键具有相同的长度*d***。 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为*d*。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是1， 我们将点积除以√d,在下面的缩放点积注意力的实现中，我们使用了暂退法进行模型正则化。

- 控制方差：点积的方差被归一化为 1，避免了方差随维度 d*d* 增大的问题。
- 稳定梯度：softmax 函数的输入值被限制在一个合理的范围内，避免了梯度消失问题。
- 提高模型性能：使注意力机制更加稳定和高效。

>
>
> 3. **点积方差过大的后果**
>
> 如果点积的方差过大（即点积的值可能非常大），softmax 的输入值 zi 也会变得非常大。这会导致以下问题：
>
> （1）**softmax 输出接近 one-hot 向量**
>
> - 当某个 zi 远大于其他 zj时，$e^{zi}$会远大于其他 $e^{zj}$。
>
> - 因此，softmax(zi) 会接近 1，而其他 softmax(zj)会接近 0。
>
> - 这意味着 softmax 的输出会接近一个 one-hot 向量，即几乎所有的概率质量集中在某一个位置。
>
>

```python
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        # queries的形状：(batch_size，查询的个数，d)
        # keys的形状：(batch_size，“键－值”对的个数，d)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

## 小结

- 注意力分数是query和key的相似度，注意力权重是分数的softmax结果

- 两种常见的分数计算

- 将query和key合并起来进入一个单输出单隐藏层的MLP
- 直接将query和key做内积

- 将注意力汇聚的输出计算可以作为值的加权平均，选择不同的注意力评分函数会带来不同的注意力汇聚操作。

- 当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高。
