---
layout: article
title: 学习文档
permalink: /learning/
sidebar:
  nav: learning
aside:
  toc: true
comment: false
show_edit_on_github: false
show_subscribe: false
sharing: false
license: false
---

本学习资料是个人的学习总结，是一套面向深度学习与大模型学习的学习资料，内容从 Python 与 PyTorch 基础开始，逐步过渡到神经网络建模、模型训练与优化、计算机视觉、自然语言处理、注意力机制、Transformer、BERT、大模型推理、Agent 系统与相关LLM面试基础等内容。

## 学习建议

本教程内容跨度较大，不建议一次性快速浏览。初学者应优先掌握神经网络、损失函数、梯度下降、模型评估和 PyTorch 实现；之后再学习 CNN、RNN、注意力机制和 Transformer。对于大模型相关章节，建议在理解 Transformer 基础后再阅读，否则容易只记住概念而难以理解其技术本质。

## 课程目录

当前共整理 29 篇学习文档。可以按章节顺序阅读，也可以使用左侧导航或站内搜索定位内容。

### 入门基础

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/01-python.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/01-python.html' | relative_url }}">01. Python 基础语法</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/02-pytorch-framework-and-functions.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/02-pytorch-framework-and-functions.html' | relative_url }}">02. PyTorch 框架</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

### 神经网络基础

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/03-linear-regression-neural-network.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/03-linear-regression-neural-network.html' | relative_url }}">03. 单层神经网络：线性回归</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/04-multilayer-neural-network.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/04-multilayer-neural-network.html' | relative_url }}">04. 多层神经网络</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/05-loss-functions.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/05-loss-functions.html' | relative_url }}">05. 神经网络损失函数</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/06-neural-network-training.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/06-neural-network-training.html' | relative_url }}">06. 神经网络学习与梯度下降</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/07-pytorch-network-implementation.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/07-pytorch-network-implementation.html' | relative_url }}">07. 深度学习基础网络搭建与实现</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/08-model-evaluation.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/08-model-evaluation.html' | relative_url }}">08. 机器学习目标与模型评估</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/09-model-fitting-and-regularization.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/09-model-fitting-and-regularization.html' | relative_url }}">09. 模型拟合度与结构调整</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/10-deep-learning-optimization-algorithms.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/10-deep-learning-optimization-algorithms.html' | relative_url }}">10. 深度学习优化算法</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/11-normalization-and-batch-normalization.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/11-normalization-and-batch-normalization.html' | relative_url }}">11. 数据归一化与 Batch Normalization</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

### NLP 与 CV 基础

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/12-computer-vision-foundations.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/12-computer-vision-foundations.html' | relative_url }}">12. 计算机视觉基础</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/13-nlp-foundations-and-word2vec.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/13-nlp-foundations-and-word2vec.html' | relative_url }}">13. NLP 基础</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>


<div class="card card--clickable" onclick="window.location.href='{{ '/learning/14-gated-recurrent-unit.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/14-gated-recurrent-unit.html' | relative_url }}">14. GRU 门控循环单元</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/15-long-short-term-memory.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/15-long-short-term-memory.html' | relative_url }}">15. LSTM 长短期记忆网络</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/16-encoder-decoder-architecture.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/16-encoder-decoder-architecture.html' | relative_url }}">16. 编码器-解码器架构</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/17-sequence-to-sequence-learning.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/17-sequence-to-sequence-learning.html' | relative_url }}">17. Seq2Seq 序列到序列学习</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/18-attention-mechanism.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/18-attention-mechanism.html' | relative_url }}">18. 注意力机制</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/19-attention-scores.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/19-attention-scores.html' | relative_url }}">19. 注意力分数</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/20-attention-based-seq2seq.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/20-attention-based-seq2seq.html' | relative_url }}">20. 使用注意力机制的 Seq2Seq</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/21-self-attention-and-positional-encoding.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/21-self-attention-and-positional-encoding.html' | relative_url }}">21. 自注意力与位置编码</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/22-transformer-architecture.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/22-transformer-architecture.html' | relative_url }}">22. Transformer 架构</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/23-bert-pretraining-and-finetuning.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/23-bert-pretraining-and-finetuning.html' | relative_url }}">23. BERT 预训练与微调</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

### 大模型相关

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/24-llm-inference-batching-and-vllm.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/24-llm-inference-batching-and-vllm.html' | relative_url }}">24. LLM 推理、Continuous Batching 与 vLLM</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>
<div class="card card--clickable" onclick="window.location.href='{{ '/learning/25-agent-systems.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/25-agent-systems.html' | relative_url }}">25. Agent 基础</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>


<div class="card card--clickable" onclick="window.location.href='{{ '/learning/26-llm-interview-basics.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/26-llm-interview-basics.html' | relative_url }}">26. LLM 面试基础</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>
<div class="card card--clickable" onclick="window.location.href='{{ '/learning/27-deep-learning-foundations.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/27-deep-learning-foundations.html' | relative_url }}">27. 其他</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

### 拓展


<div class="card card--clickable" onclick="window.location.href='{{ '/learning/28-reinforcement-learning.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/28-reinforcement-learning.html' | relative_url }}">28. 强化学习</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>

<div class="card card--clickable" onclick="window.location.href='{{ '/learning/29-web-crawler.html' | relative_url }}'" style="margin-bottom:1rem;">
  <div class="card__content">
    <h3><a href="{{ '/learning/29-web-crawler.html' | relative_url }}">29. 网络爬虫基础</a></h3>
    <p>点击进入本章节学习内容。</p>
  </div>
</div>
