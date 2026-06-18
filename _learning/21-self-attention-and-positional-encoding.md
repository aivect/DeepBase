---
layout: article
title: "自注意力与位置编码"
key: learning-21-self-attention-and-positional-encoding
permalink: /learning/21-self-attention-and-positional-encoding.html
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

## 1 自注意selfAttention

### 1.1 自注意 selfAttention

$$
\begin{aligned}
\bullet\;&\text{给定序列 }\mathbf{x}_1,\ldots,\mathbf{x}_n,\quad \mathbf{x}_i\in\mathbb{R}^d \\
\bullet\;&\text{自注意力池化层将 }\mathbf{x}_i\text{ 同时作为 query、key 和 value，对序列抽取特征，得到 }\mathbf{y}_1,\ldots,\mathbf{y}_n \\
&\mathbf{y}_{i}= f\bigl(\mathbf{x}_{i},(\mathbf{x}_{1},\mathbf{x}_{1}),\ldots,(\mathbf{x}_{n},\mathbf{x}_{n})\bigr)\in\mathbb{R}^{d}
\end{aligned}
$$

<img src="{{ '/assets/learning/21_self_attention_and_positional_encoding/figure_003.png' | relative_url }}" alt="image-20230831231841457" style="zoom:50%;" />

### 1.2 跟CNN，RNN对比

<img src="{{ '/assets/learning/21_self_attention_and_positional_encoding/figure_004.png' | relative_url }}" alt="image-20230831232336064" style="zoom: 50%;" />

n是时间步大小，d是dimension

### 1.3 位置编码

- 跟 CNN/RNN 不同，自注意力并没有记录位置信息
- 位置编码将位置信息注入到输入里
- $\begin{aligned}&\text{假设长度为 }n\text{ 的序列是 }\mathbf{X}\in\mathbb{R}^{n\times d}\text{,那么使用位置编码矩阵}\\&\mathbf{P}\in\mathbb{R}^{n\times d}\text{ 来输出 }\mathbf{X}+\mathbf{P}\text{ 作为自编码输入}\end{aligned}$

- $\begin{aligned}&\mathbf{P}\text{ 的元素如下计算:}\\&p_{i,2j}=\sin\left(\frac i{10000^{2j/d}}\right),\quad p_{i,2j+1}=\cos\left(\frac i{10000^{2j/d}}\right)\end{aligned}$

𝑑代表这个模型中此时输入向量的维度。

<img src="{{ '/assets/learning/21_self_attention_and_positional_encoding/figure_001.png' | relative_url }}" alt="image-20230831233307861" style="zoom:50%;" />

### 1.4 相对位置信息

<img src="{{ '/assets/learning/21_self_attention_and_positional_encoding/figure_002.png' | relative_url }}" alt="image-20230901093111339" style="zoom:50%;" />

### 1.5 总结

- 自注意力池化层将x$i$当做key，value,query来对序列抽取特征
- 完全并行、最长序列为1、但对长序列计算复杂度高
- 位置编码在输入中加入位置信息，使得自注意力能够记忆位置信息

## 2 代码

位置编码

```python
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))#num_hiddens特征维度
        X = torch.arange(max_len, dtype=torch.float32).reshape(
        -1, 1) / torch.pow(10000, torch.arange(
        0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    #对应位置编码公式
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)#取出对应长度的位置编码
        return self.dropout(X)#dropout避免对P太过敏感
```

---

##  LLaMA Rotary Position Embedding

参考 [https://mp.weixin.qq.com/s/0peSNWN0ypMopPR0Q\_pujQ](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/0peSNWN0ypMopPR0Q_pujQ)

