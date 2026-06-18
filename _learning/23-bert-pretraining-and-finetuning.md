---
layout: article
title: "BERT 预训练与微调"
key: learning-23-bert-pretraining-and-finetuning
permalink: /learning/23-bert-pretraining-and-finetuning.html
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

参考：

- [读懂 BERT，看这一篇就够了](https://zhuanlan.zhihu.com/p/403495863)

- [BERT 的嵌入层是如何实现的](https://mp.weixin.qq.com/s/DfIAuo775_sHGYi5z9IZyw)

## 1 BERT嵌入

Token嵌入：

BERT 使用WordPiece 作为分词方法，将单词划分为子词单元。token嵌入层的作用是将单词转换为固定维的向量表示形式。在 BERT 的例子中，每个单词都表示为一个768维的向量。

Segment嵌入：

Segment嵌入层只有两个向量表示。第一个向量(索引0)分配给属于输入1的所有tokens，而最后一个向量(索引1)分配给属于输入2的所有tokens。如果一个输入只有一个输入语句，那么它的Segment嵌入就是对应于Segment嵌入表的索引为0的向量。

Position嵌入：

BERT 被设计用来处理长度为512的输入序列。作者通过让 BERT 学习每个位置的向量表示来包含输入序列的顺序特征。这意味着Position嵌入层是一个大小为(512,768)的查找表，其中第一行是第一个位置上的任意单词的向量表示，第二行是第二个位置上的任意单词的向量表示。

- 在 BERT 的MLM任务中，**一个`[MASK]`标记确实对应一个被遮蔽的token**；
- BERT 的预训练中，**MLM和NSP是同时训练的**
- MLM和NSP任务实现方式
- 共享Encoder：两个任务共用同一Encoder，但分别通过不同的输出层处理：
- MLM任务：对遮蔽位置的Encoder输出接一个**词汇表大小的分类器**（预测被遮蔽词）。（线性层（映射到对应的词汇数量） + Softmax）
- NSP任务：对[CLS]标记的Encoder输出接一个**二分类器**（预测句子关系）。
联合训练：两者的损失函数加权求和，同步优化Encoder参数。（**线性层 + Softmax**：在 BERT 的Encoder输出后，接一个全连接层（`Linear`）将`[CLS]`标记的768维向量（以 BERT-Base为例）映射到2维输出（对应二分类的类别数））

**BERT 本质上是在海量语料的基础上，通过自监督学习的方法为单词学习一个好的特征表示。**该模型的优点是
可以根据具体的人物进行微调，或者直接使用预训练的模型作为特征提取器。

## 2 BERT的动机

- 基于微调的 NLP 模型
- 预训练的模型抽取了足够多的信息
- 新的任务只需要增加一个简单地输出层

<img src="{{ '/assets/learning/23_bert_pretraining_and_finetuning/figure_003.png' | relative_url }}" alt="image-20230902152703390" style="zoom:50%;" />

## 3 BERT架构

- 只有编码器的 Transformer
- 两个版本：
- Base:#blocks=12,hidden size=768,#heads=12,#parameters=110M
- Large:#blocks=24,hidden size=1024,#heads=16,#paramerter=340M

- 在大规模数据上训练>3B词

## 4 对输入的修改

- 每个样本是一个句子对
- 加入额外的片段嵌入
- 位置编码可学习，不再使用sin、cos

![image-20230902153228658]({{ '/assets/learning/23_bert_pretraining_and_finetuning/figure_004.png' | relative_url }})

## 5 预训练任务

### 5.1 预训练任务1：带掩码的语言模型

不是预测未来，我看双向信息，加一些<mask>，看见<mask>就根据上下文做完形填空（预测）是可以的。

- Transformer 的编码器是双向的，标准语言模型要求单向
- 带掩码的语言模型每次随机（15%概率）将一些词元换成<mask>
- 因为微调任务中不出现<mask>
- 80%概率下，将选中的词元变成<mask>
- 10%概率下换成一个随机词元。
- 10%概率下保持原有的词元

### 5.2 预训练任务2：下一个句子预测

- 预测一个句子对中两个句子是不是相邻
- 训练样本中：
- 正：50%概率选择相邻句子对：<cls>this movie is great <sep> i like it <sep>
- 负：50%概率选择随机句子对：<cls>this movie is great<sep> hello world<sep>

- 将<cls>对应的输出放到一个全连接层来预测，预测这两个句子是不是相邻的

## 6 代码实现

### 6.1 获取输入

在自然语言处理中，有些任务（如情感分析）以单个文本作为输入，而有些任务（如自然语言推断）以一对文本序列作为输入。BERT 输入序列明确地表示单个文本和文本对。当输入为单个文本时，BERT 输入序列是特殊类别词元“<cls>”、文本序列的标记、以及特殊分隔词元“<sep>”的连结。当输入为文本对时，BERT 输入序列是“<cls>”、第一个文本序列的标记、“<sep>”、第二个文本序列标记、以及“<sep>”的连结。我们将始终如一地将术语“BERT 输入序列”与其他类型的“序列”区分开来。例如，一个*BERT 输入序列*可以包括一个*文本序列*或两个*文本序列*。

为了区分文本对，根据输入序列学到的片段嵌入eA和eB分别被添加到第一序列和第二序列的词元嵌入中。对于单文本输入，仅使用eA。

下面的`get_tokens_and_segments`将一个句子或两个句子作为输入，然后返回 BERT 输入序列的标记及其相应的片段索引。

```python
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
    tokens += tokens_b + ['<sep>']
    segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT 选择 Transformer 编码器作为其双向架构。在 Transformer 编码器中常见是，位置嵌入被加入到输入序列的每个位置。然而，与原始的 Transformer 编码器不同，BERT 使用*可学习的*位置嵌入。总之， 下图表明 BERT 输入序列的嵌入是词元嵌入、片段嵌入和位置嵌入的和。

![image-20230902162121596]({{ '/assets/learning/23_bert_pretraining_and_finetuning/figure_004.png' | relative_url }})

Segment嵌入层只有两个向量表示。

### 6.2 BERT实现

下面的`BERTEncoder`类类似于 [10.7节](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html#sec-transformer)中实现的`TransformerEncoder`类。与`TransformerEncoder`不同，`BERTEncoder`使用片段嵌入和可学习的位置嵌入。

```python
#@save
class BERTEncoder(nn.Module):
    """BERT 编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
    ffn_num_hiddens, num_heads, num_layers, dropout,
    max_len=1000, key_size=768, query_size=768, value_size=768,
    **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
        self.blks.add_module(f"{i}", d2l.EncoderBlock(
        key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在 BERT 中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
        X = blk(X, valid_lens)
        return X
```

假设词表大小为10000，为了演示`BERTEncoder`的前向推断，让我们创建一个实例并初始化它的参数。

```python
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
 ffn_num_hiddens, num_heads, num_layers, dropout)
```

我们将`tokens`定义为长度为8的2个输入序列，其中每个词元是词表的索引。使用输入`tokens`的`BERTEncoder`的前向推断返回编码结果，其中每个词元由向量表示，其长度由超参数`num_hiddens`定义。此超参数通常称为 Transformer 编码器的*隐藏大小*（隐藏单元数）

```python
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

### 6.3 预训练任务

#### 6.3.1 遮掩语言模型（Masked Language Model, MLM）

我们实现了下面的`MaskLM`类来预测 BERT 预训练的掩蔽语言模型任务中的掩蔽标记。预测使用单隐藏层的多层感知机（`self.mlp`）。在前向推断中，它需要两个输入：`BERTEncoder`的编码结果和用于预测的词元位置。输出是这些位置的预测结果。

```python
#@save
class MaskLM(nn.Module):
    """BERT 的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),nn.ReLU(),nn.LayerNorm(num_hiddens),
        nn.Linear(num_hiddens, vocab_size))

    #pred_position是待预测的<mask>的位置
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

为了演示`MaskLM`的前向推断，我们创建了其实例`mlm`并对其进行了初始化。回想一下，来自`BERTEncoder`的正向推断`encoded_X`表示2个 BERT 输入序列。我们将`mlm_positions`定义为在`encoded_X`的任一输入序列中预测的3个指示。`mlm`的前向推断返回`encoded_X`的所有掩蔽位置`mlm_positions`处的预测结果`mlm_Y_hat`。对于每个预测，结果的大小等于词表的大小。

```python
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

通过掩码下的预测词元`mlm_Y`的真实标签`mlm_Y_hat`，我们可以计算在 BERT 预训练中的遮蔽语言模型任务的交叉熵损失。

```python
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

#### 6.3.2 下一句预测（Next Sentence Prediction, NSP）

下面的`NextSentencePred`类使用单隐藏层的多层感知机来预测第二个句子是否是 BERT 输入序列中第一个句子的下一个句子。由于 Transformer 编码器中的自注意力，特殊词元“<cls>”的 BERT 表示已经对输入的两个句子进行了编码。因此，多层感知机分类器的输出层（`self.output`）以`X`作为输入，其中`X`是多层感知机隐藏层的输出，而MLP隐藏层的输入是编码后的“<cls>”词元。

```python
#@save
class NextSentencePred(nn.Module):
    """BERT 的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
```

我们可以看到，`NextSentencePred`实例的前向推断返回每个 BERT 输入序列的二分类预测。

```python
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

### 6.4 整合代码

在预训练 BERT 时，最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合。现在我们可以通过实例化三个类`BERTEncoder`、`MaskLM`和`NextSentencePred`来定义`BERTModel`类。前向推断返回编码后的 BERT 表示`encoded_X`、掩蔽语言模型预测`mlm_Y_hat`和下一句预测`nsp_Y_hat`。

```python
#@save
class BERTModel(nn.Module):
    """BERT 模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
    ffn_num_hiddens, num_heads, num_layers, dropout,
    max_len=1000, key_size=768, query_size=768, value_size=768,
    hid_in_features=768, mlm_in_features=768,
    nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
        ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
        dropout, max_len=max_len, key_size=key_size,
        query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
        nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
    pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
        mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
        mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```



## 7 总结

- BERT 针对 NLP 的微调设计（针对不同任务，改改输出层，以及权重等）
- 基于 Transformer 的编码器做了如下修改
- 模型更大，训练数据更多
- 输入句子对，片段嵌入，可学习的位置编码
- 训练时使用两个任务：
- 带掩码的语言模型
- 下一个句子预测

![image-20230905191439396]({{ '/assets/learning/23_bert_pretraining_and_finetuning/figure_001.png' | relative_url }})

### 

## 8 BERT微调

与图片分类不同，BERT 预训练时使用的两个任务没有什么实际应用场景，所以使用 BERT 时多需要进行微调。

BERT 对每一个token都返回一个特定长度的特征向量（课堂演示为128，bert-base是768，bert-large是1024），这些特征向量抽取了上下文信息。不同的任务使用不同的特征。

### 8.1 具体应用

#### （1）句子分类

将句首的\<CLS\>token对应的向量输入到全连接层分类。对于一对句子也是同理，句子中间用\<SEP\>分开但仍只用第一个\<CLS\>对应的向量。

关于为什么要使用\<CLS\>是因为预训练中判断句子是否连续任务中使用的是\<CLS\>，因此模型会“知道”\<CLS\>是句子级别分类用的向量，表示的信息应与句子整体有关。当然我们也可以不使用\<CLS\>选定自己想要的token，之后在微调中更新bert的权重即可。

#### （2）命名实体识别

命名实体识别即识别一个词元是不是命名实体，例如人名、机构、位置。其方法是将每一个非特殊词元的向量放进全连接层分类（二分类多分类均可）。

#### （3） 问题回答

**给定一个问题和描述文字，找出一个判断作为回答，微调方法为对片段中的每个词元预测它是不是回答的开头或结束。**

总体而言，无论是句子级别还是词级别的分类任务，都只需要在bert的基础上加全连接层，bert中的权重是可以直接从预训练模型得到的，真正需要自己从头训练的只有全连接层的权重。
