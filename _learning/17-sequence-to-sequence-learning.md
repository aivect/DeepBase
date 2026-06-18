---
layout: article
title: "Seq2Seq 序列到序列学习"
key: learning-17-sequence-to-sequence-learning
permalink: /learning/17-sequence-to-sequence-learning.html
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

## 1 序列到序列学习（训练一个序列句子，一个batch）

## 1 应用举例：机器翻译

- 给定一个源语言的句子，自动翻译成目标语言
- 这两个句子可以有不同的长度

### 2. 模型架构：Seq2seq

![image-20230828210300226]({{ '/assets/learning/17_sequence_to_sequence_learning/figure_001.png' | relative_url }})

- 序列到序列模型由**编码器-解码器**构成。

- **编码器**RNN 可以是**双向**，由于输入的句子是完整地，可以正着看，也可以反着看；而**解码器**只能是**单向**，由于预测时，只能正着去预测。

- 编码器，解码器采用**不同的 RNN**，此 RNN 也可以是 GRU（门控循环单元），LSTM（长短期记忆网络）等。

LSTM、GRU、Seq2Seq 的区别是什么？

> Seq2Seq 是一种由编码器和解码器组成的框架，而 LSTM、GRU 是组成编码器和解码器的一种单元。

## 3. 编码器-解码器细节

![image-20230828210347092]({{ '/assets/learning/17_sequence_to_sequence_learning/figure_003.png' | relative_url }})

- 编码器的 RNN**没有**连接**输出层**

- **编码器**的**最后时间步的隐状态**用作**解码器**的**初始隐状态**（图中箭头的传递）

## 4. 训练和推理

![image-20230828210448404]({{ '/assets/learning/17_sequence_to_sequence_learning/figure_002.png' | relative_url }})

- 第3节中提到编码器没有输出层，只有解码器有，于是损失函数的计算只关注解码器的输出层。
- 训练和预测（推理）有区别的，训练时解码器使用目标句子（真值）作为输入，以指导模型训练；而推理时无法提前得知真值，需要一步一步进行预测。

## 2 代码

现在，让我们[**实现循环神经网络编码器**]。 注意，我们使用了*嵌入层*（embedding layer） 来获得输入序列中每个词元的特征向量。 嵌入层的权重是一个矩阵， 其行数等于输入词表的大小（`vocab_size`）， 其列数等于特征向量的维度（`embed_size`）。 对于任意输入词元的索引$i$，
嵌入层获取权重矩阵的第$i$行（从$0$开始）以返回其特征向量。
另外，本文选择了**一个多层门控循环单元**来实现编码器。

### 2.1 Encoder

从技术上讲，编码器将长度可变的输入序列转换成
形状固定的上下文变量$\mathbf{c}$，
并且将输入序列的信息在该上下文变量中进行编码。
如 :numref:`fig_seq2seq`所示，可以使用循环神经网络来设计编码器。

考虑由一个序列组成的样本（批量大小是$1$）。
假设输入序列是$x_1, \ldots, x_T$，
其中$x_t$是输入文本序列中的第$t$个词元。
在时间步$t$，循环神经网络将词元$x_t$的输入特征向量
$\mathbf{x}_t$和$\mathbf{h} _{t-1}$（即上一时间步的隐状态）
转换为$\mathbf{h}_t$（即当前步的隐状态）。
使用一个函数$f$来描述循环神经网络的循环层所做的变换：

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

总之，编码器通过选定的函数$q$，
将所有时间步的隐状态转换为上下文变量：

$$\mathbf{c} = q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

比如，当选择$q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$时
（就像 :numref:`fig_seq2seq`中一样），
上下文变量仅仅是输入序列在最后时间步的隐状态$\mathbf{h}_T$。

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d21
```

num_layers:多少个 RNN 层

```python
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```

实例化：

```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
# torch.Size([7, 4, 16])
```

### 2.2 Decoder

广播context，使其具有与X相同的num_steps,再与encoder的输入在dim=2（即列上）,即dimension做cat。

```python
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 返回encoder中的hn
        return enc_outputs[1]

    def forward(self, X, state):
        # x -->(4,7)
        # state-->(7,4,16)
        # 输出新'X'的形状：(num_steps,batch_size,embed_size)-->(7,4,8)
        X = self.embedding(X).permute(1, 0, 2)
        # 1将最后一个时间步在最后一个 RNN 隐藏层的隐藏状态hn拿出，这个基本含有前面所有信息
        # 2广播context，使其具有与X相同的num_steps,再与encoder的输入在dim=2（即列上）,即dimension做cat。
        # 因此在nn.GRU(embed_size + num_hiddens,……）为embed_size + num_hiddens

        #context：state[-1]-->(1,4,16)-->.repeat(X.shape[0], 1, 1)-->(7,4,16)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # torch.cat((X, context), 2)-->(7,4,8+16)
        X_and_context = torch.cat((X, context), 2)
        #output-->(7,4,16), state(2,4,16)
        output, state = self.rnn(X_and_context, state)
        # output -->(4,7,10)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```

实例化：

```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
encoder.eval()
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
# encoder(X)-->output(7,4,16),state(1,4,26)
# decoder.init_state(encoder(X))-->(7,4,16)
# decoder(X, state)-->
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)-->
output.shape, state.shape
# (torch.Size([4, 7, 10]), torch.Size([2, 4, 16]))
```

### 2.3 损失函数

**在每个时间步，解码器预测了输出词元的概率分布**。
类似于语言模型，可以使用softmax来获得分布，
并通过计算交叉熵损失函数来进行优化。
回想一下 :numref:`sec_machine_translation`中，
特定的填充词元被添加到序列的末尾，
因此不同长度的序列可以以相同形状的小批量加载。
但是，我们应该将填充词元的预测排除在损失函数的计算之外。

为此，我们**可以使用下面的`sequence_mask`函数**
**[通过零值化屏蔽不相关的项]**，
以便后面任何不相关预测的计算都是与零的乘积，结果都等于零。
例如，如果两个序列的有效长度（不包括填充词元）分别为$1$和$2$，
则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为零。

```python
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
    device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
'''
tensor([[1, 0, 0],
 [4, 5, 0]])'''
```

- 使用 `~mask` 对布尔张量 `mask` 进行**逐元素逻辑取反**的操作，对掩码取反，得到需要屏蔽的位置。
- 将 `X` 中对应 `~mask` 为 `True` 的位置设置为 `value`。

现在，我们可以[**通过扩展softmax交叉熵损失函数来遮蔽不相关的预测**]。
最初，所有预测词元的掩码都设置为1。
一旦给定了有效长度，与填充词元对应的掩码将被设置为0。
最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。

在 PyTorch 中使用`CrossEntropyLoss`时，将标签设置为`-100`是一种**屏蔽特定样本损失**的有效方法：

- PyTorch 的`CrossEntropyLoss`（以及`NLLLoss`）对`target`张量中的`-100`值有特殊处理：

- **自动忽略**：标签为`-100`的样本不参与损失计算
- **梯度传播**：对应位置的梯度为0，不影响反向传播
- **batch维度保持**：仅减少有效样本数，不改变输出张量形状

```
 # 序列标注任务示例 (pad_token_id = -100)
 labels = torch.tensor([
 [5, 2, 3, 0],
 [1, 4, -100, -100], # 第2行的后两个token被屏蔽
 [2, 6, 1, -100] # 最后一列的token被屏蔽
 ])

 logits = torch.randn(3, 4, 5) # (batch, seq_len, num_classes)
 loss = criterion(logits.view(-1, 5), labels.view(-1))
```

```python
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
 """带遮蔽的softmax交叉熵损失函数"""
 # pred的形状：(batch_size,num_steps,vocab_size)
 # label的形状：(batch_size,num_steps)
 # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

将 `nn.CrossEntropyLoss` 的 `reduction` 参数设置为 `'none'`，表示不进行任何规约降维操作（即不求和或求平均），直接返回每个位置的损失值。这样，`unweighted_loss` 的形状将与 `label` 相同，为 `(batch_size, num_steps)`。

`nn.CrossEntropyLoss` 对输入张量的形状有明确的要求，具体如下：

输入要求：

- **预测值（input）**：形状应为 `(batch_size, num_classes, *)`，其中 `*` 表示任意数量的附加维度（如序列长度）。
例如，对于序列数据，预测值的形状应为 `(batch_size, num_classes, seq_len)`。
- **目标（target）**：形状应为 `(batch_size, *)`，即与预测值的非类别维度（如序列长度）匹配。
例如，对于序列数据，目标的形状应为 `(batch_size, seq_len)`，且每个元素是类别索引（从 0 开始）。

为什么需要 `permute(0, 2, 1)`：

如果你从模型中得到的预测值形状是 `(batch_size, seq_len, num_classes)`（例如，某些 RNN 或 Transformer 模型的输出），而 `nn.CrossEntropyLoss` 要求 `(batch_size, num_classes, seq_len)`，那么你需要手动交换维度：

```python
pred = pred.permute(0, 2, 1) # 从 (B, S, C) -> (B, C, S)
loss = nn.CrossEntropyLoss()(pred, label)
```

`nn.CrossEntropyLoss` 的内部实现：

1. 对输入张量的最后一个维度（即类别维度）应用 `LogSoftmax`。
2. 计算该维度与目标张量的交叉熵损失。
它不会对其他维度进行任何形状调整，因此必须确保输入张量的形状符合要求。

我们可以创建三个相同的序列来进行[**代码健全性检查**]，
然后分别指定这些序列的有效长度为$4$、$2$和$0$。
结果就是，第一个序列的损失应为第二个序列的两倍，而第三个序列的损失应为零。

```python
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),torch.tensor([4, 2, 0]))
# tensor([2.3026, 1.1513, 0.0000])
```

### 2.4 训练

在下面的循环训练过程中，如 :numref:`fig_seq2seq`所示， **特定的序列开始词元（“<bos>”）和 原始的输出序列（不包括序列结束词元“<eos>”） 拼接在一起作为解码器的输入。 这被称为*核心理解*（teacher forcing）**， 因为原始的输出序列（词元的标签）被送入解码器。 或者，将来自上一个时间步的*预测*得到的词元作为解码器的当前输入。

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if type(m) nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])

                        net.apply(xavier_init_weights)
                        net.to(device)
                        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                        loss = MaskedSoftmaxCELoss()
                        net.train()
                        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                                xlim=[10, num_epochs])
                        for epoch in range(num_epochs):
                            timer = d2l.Timer()
                            metric = d2l.Accumulator(2) # 训练损失总和，词元数量
                            for batch in data_iter:
                                optimizer.zero_grad()
                                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                                                   device=device).reshape(-1, 1)
                                dec_input = torch.cat([bos, Y[:, :-1]], 1) # 核心理解
                                Y_hat, _ = net(X, dec_input, X_valid_len)
                                l = loss(Y_hat, Y, Y_valid_len)
                                l.sum().backward()	# 损失函数的标量进行“反向传播”
                                d2l.grad_clipping(net, 1)
                                num_tokens = Y_valid_len.sum()
                                optimizer.step()
                                with torch.no_grad():
                                    metric.add(l.sum(), num_tokens)
                                    if (epoch + 1) % 10 0:
                                        animator.add(epoch + 1, (metric[0] / metric[1],))
                                        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
                                              f'tokens/sec on {str(device)}')
```

### 2.5 预测

为了采用一个接着一个词元的方式预测输出序列，
每个解码器当前时间步的输入都将来自于前一时间步的预测词元。
与训练类似，序列开始词元（“&lt;bos&gt; begin of sentence”）
在初始时间步被输入到解码器中。
该预测过程如 :numref:`fig_seq2seq_predict`所示，
当输出序列的预测遇到序列结束词元（“&lt;eos&gt;”）时，预测就结束了。
我们将在 :numref:`sec_beam-search`中介绍不同的序列生成策略。

> 添加'<eos>'和'<pad>'标记是为了在预测时生成完整的句子，并保持输入序列的长度一致。而在训练时，由于目标序列已知，有eos，且等长，不需要这些额外的标记。

```python
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    # 这里在进入解码器前面加上'<eos>'，并填充到num_steps长度，目的在于使预测时能够很好的获取到结束位置信息
    #这与训练时不同，训练数据时间步相同，不需要填充，
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    # num_steps是时间步大小
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state) ### 【注意！！！】
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights: attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred tgt_vocab['<eos>']:
            break
            output_seq.append(pred)
            return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

### 2.6 预测序列的评估BLEU

```python
def bleu(pred_seq, label_seq, k): #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
            for i in range(len_pred - n + 1):
                if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                    num_matches += 1
                    label_subs[' '.join(pred_tokens[i: i + n])] -= 1
                    score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
                    return score
```

```python
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

    '''go . => va tomber !, bleu 0.000
i lost . => j'ai <unk> ., bleu 0.000
he's calm . => assieds-toi ici ., bleu 0.000
i'm home . => je suis chez paresseux ., bleu 0.752'''
```

### 总结

* 根据“编码器-解码器”架构的设计，
我们可以使用两个循环神经网络来设计一个序列到序列学习的模型。
* 在实现编码器和解码器时，我们可以使用多层循环神经网络。
* 我们可以使用遮蔽来过滤不相关的计算，例如在计算损失时。
* 在“编码器－解码器”训练中，核心理解方法将原始输出序列（而非预测结果）输入解码器。
* BLEU是一种常用的评估方法，它通过测量预测序列和标签序列之间的$n$元语法的匹配度来评估预测。
