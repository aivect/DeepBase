---
layout: article
title: "编码器-解码器架构"
key: learning-16-encoder-decoder-architecture
permalink: /learning/16-encoder-decoder-architecture.html
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

## 1.1 CNN中的解释

考虑一个 CNN 模型：

![CNN]({{ '/assets/learning/16_encoder_decoder_architecture/figure_004.png' | relative_url }})

整个 CNN 实际上可以看作一个编码器，解码器两部分。

* 底层的神经网络，也就是编码器将输入编码成能被模型识别的中间表达形式，也就是特征
* 解码器将中间结果解码为输出

## 1.2 RNN中的解释

对于 RNN 而言，同样有着类似的划分

![RNN]({{ '/assets/learning/16_encoder_decoder_architecture/figure_001.png' | relative_url }})

* 编码器将输入文本表示为向量
* 解码器将向量表示为输出

### 1.3 抽象的编码器-解码器架构

指一个模型被分为两块：

* 一块是编码器，也叫encoder，用于将输入处理为一个**中间状态**
* 一块是解码器，也叫decoder，用于将**中间状态表示为输出**
* **解码器也可以有额外的输入提供信息**

![encoder-decoder]({{ '/assets/learning/16_encoder_decoder_architecture/figure_002.png' | relative_url }})

![image-20230828212721192]({{ '/assets/learning/16_encoder_decoder_architecture/figure_003.png' | relative_url }})

- 文本处理和语音识别的Encoder部分通常采用 RNN 模型
- 图像处理的Encoder一般采用 CNN 模型

## 2 代码

### 2.1 Encoder

```python
from torch import nn
class Encoder(nn.Module)
    '''编码器-解码器结构的基本编码器接口。'''
    def __init__ (self, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        def forward(self,X, *args):
            raise NotImplementedError
```

### 2.2 Decoder

```python
class Decoder(nn.Module):
    '''编码器-解码器结构的基本解码器接口。'''
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    def ini_state(self, enc_routputs, *args):
        raise NotImplementedError
    def forward(self, x,state):
        raise NotImplementedError
```

> ini_state(self, enc_routputs, *args):中的**enc_routputs是Encoder的 RNN 层的output和 hn**

### 2.3 合并Encoder和Decoder

```python
class EncoderDecoder(nn.Module):
    '''编码器-解码器结构的基'''
    def __init_(self, encoder,decoder, **kwargs):
        super(EncoderDecoder,self).__init__ (**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_ X,*args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs， *args)
        return self.decoder(dec_X，dec_state)

```
