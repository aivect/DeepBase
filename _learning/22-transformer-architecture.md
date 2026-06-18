---
layout: article
title: "Transformer 架构"
key: learning-22-transformer-architecture
permalink: /learning/22-transformer-architecture.html
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

参考：[Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)

## 1 Transformer 的整体结构

Transformer 是一种基于注意力机制的序列建模架构，最初用于机器翻译任务。与 RNN 按时间步递归处理序列不同，Transformer 通过自注意力机制并行建模序列内部的依赖关系，因此更适合大规模并行训练。

<img src="{{ '/assets/learning/22_transformer_architecture/figure_005.png' | relative_url }}" alt="Transformer 编码器-解码器整体结构" style="zoom: 50%;" />

整体上，Transformer 由编码器（Encoder）和解码器（Decoder）组成：编码器负责将输入序列编码为上下文表示；解码器在已生成目标序列的基础上，结合编码器输出逐步生成下一个 token。

## 2 自注意力机制

自注意力的核心思想是：序列中每个 token 都可以根据与其他 token 的相关性，动态聚合上下文信息。给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$，缩放点积注意力定义为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $d_k$ 是 key 的维度。除以 $\sqrt{d_k}$ 的目的是避免点积值随维度增大而过大，从而导致 softmax 梯度过小。

<img src="{{ '/assets/learning/22_transformer_architecture/figure_007.png' | relative_url }}" alt="自注意力计算示意" style="zoom: 50%;" />

## 3 多头注意力

单个注意力头只能在一个表示子空间中建模相关性。多头注意力将 $Q,K,V$ 投影到多个子空间中分别计算注意力，再将结果拼接并线性变换：

$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h)W^O
$$

$$
head_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

多头机制使模型可以同时关注不同类型的关系，例如局部依赖、长距离依赖和不同语义角色之间的联系。

<img src="{{ '/assets/learning/22_transformer_architecture/figure_013.png' | relative_url }}" alt="多头注意力示意" style="zoom:50%;" />

## 4 位置编码

由于自注意力本身不包含顺序信息，Transformer 需要额外加入位置编码。常见的正弦位置编码为：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

位置编码与词向量相加后输入模型，使模型能够区分相同 token 出现在不同位置时的含义差异。

<img src="{{ '/assets/learning/22_transformer_architecture/figure_012.png' | relative_url }}" alt="位置编码示意" style="zoom:50%;" />

## 5 解码器中的 Masked Self-Attention

在生成任务中，解码器不能看到未来 token，因此需要使用因果掩码（causal mask）。掩码会阻止当前位置关注后续位置，保证训练阶段与推理阶段的一致性。

<img src="{{ '/assets/learning/22_transformer_architecture/figure_009.png' | relative_url }}" alt="Mask 矩阵示意" style="zoom:50%;" />

## 6 残差连接、Layer Normalization 与前馈网络

Transformer 的每个子层通常包含残差连接和归一化层。残差连接有助于缓解深层网络的梯度消失问题，Layer Normalization 用于稳定隐藏状态分布。前馈网络（Feed Forward Network, FFN）对每个位置独立进行非线性变换，增强模型表达能力。

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

## 7 小结

Transformer 的关键在于用注意力机制替代循环结构，通过多头注意力、位置编码、残差连接、归一化和前馈网络共同完成序列建模。编码器更强调输入序列的上下文理解，解码器则在因果掩码约束下完成自回归生成。
