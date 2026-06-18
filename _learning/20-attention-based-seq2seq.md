---
layout: article
title: "使用注意力机制的 Seq2Seq"
key: learning-20-attention-based-seq2seq
permalink: /learning/20-attention-based-seq2seq.html
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

## 1 使用注意力机制的seq2seq

### 1.1 动机

机器翻译中，每个生成的词可能相关于源句子中不同的词

<img src="{{ '/assets/learning/20_attention_based_seq2seq/figure_001.png' | relative_url }}" alt="image-20230828205217955" style="zoom:50%;" />

seq2seq模型中不能对此直接建模

### 1.2 加入注意力

![image-20230828205243529]({{ '/assets/learning/20_attention_based_seq2seq/figure_006.png' | relative_url }})

- 编码器对每次词的输出作为key和value(他俩一样，key和value就是在编码器对每个词的输出)，然后放入attention中
- 解码器 RNN 对上一个词的输出是query（query就是解码器对上一个词的预测的输出），会再次进入attention
- 注意力的输出query和下一个词的词嵌入合并进入解码器 RNN

不加入注意力：

![image-20230828213410631]({{ '/assets/learning/20_attention_based_seq2seq/figure_002.png' | relative_url }})

RNN 的输出都是在同一个语义空间。

总结：

- 1、不加入注意力，则解码器的输入永远是上一个 RNN 的最后一个词（时间步）的输出
- 2、加入注意力机制后，得到的改进则解码器 RNN 的输入是可以先是上一个 RNN 最前面作为输出，然后再逐步到后面做为输出，通过加权方式。
- 3、**Seq2seq中通过最后一个词（时间步）隐状态在编码器和解码器中传递信息**
- 4、注意力机制可以根据**解码器 RNN 的输出**来匹配到合适的**编码器 RNN 的输出**来更有效的传递信息

![image-20230904091719084]({{ '/assets/learning/20_attention_based_seq2seq/figure_003.png' | relative_url }})

![image-20230904092005168]({{ '/assets/learning/20_attention_based_seq2seq/figure_004.png' | relative_url }})
