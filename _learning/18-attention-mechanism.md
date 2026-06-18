---
layout: article
title: "注意力机制"
key: learning-18-attention-mechanism
permalink: /learning/18-attention-mechanism.html
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

心理学认为人通过随意线索和不随意线索选择注意点。

卷积、全连接、池化层都只考虑不随意线索。

注意力机制则显示的考虑随意线索

- **随意线索**被称之为查询 (query)
- 每个输入是一个值 (value) 和**不随意线索(key)**的对
- 通过注意力池化层来有偏向性的选择选择某些输入

query，也就是随意线索是人为给定的，CNN 在进行卷积之前需要按照给定的query对输入进行搜索

## 非参注意力池化层

![image-20230828104056892]({{ '/assets/learning/18_attention_mechanism/figure_003.png' | relative_url }})

- $K$就是核数

___

## Nadaraya-Watson核回归

- 使用高斯核
$$K(u) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{u^2}{2}\right)$$

- 那么
$$
\begin{aligned}
f(x) &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\
&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i
\end{aligned}
$$

---

## 参数化的注意力机制
- 在之前基础上引入可以学习的 $w$

$$
f(x) = \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}\big((x - x_i)w\big)^2\right) y_i
$$





## 总结：

心理学认为人通过随意线索和不随意线索选择注意点

注意力机制中，通过query (随意线索) 和key (不随意线索)来有偏向性的对value选择输入
$$
\text{可以一般的写作}f(x)=\sum_i\alpha(x,x_i)\nu_i,\text{,这里}\alpha(x,x_i)\text{ 是注意力权重}
$$
早在60年代就有非参数的注意力机制

注意力机制绝对是提高效率的一种必要手段，将[注意力]集中到有用的信息上，不要在噪声中花费时间。

注意力的核心目标：从众**多信息中选择出对当前任务目标更关键的信息将注意力放在这上面**。
