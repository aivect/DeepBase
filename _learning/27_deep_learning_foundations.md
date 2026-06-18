---
layout: article
title: "深度学习基础概念"
key: learning-27-deep-learning-foundations
permalink: /learning/27-deep-learning-foundations.html
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

## 1.1 加深网络的目的

加深网络的目的并不仅仅是为了更好地拟合目标函数，而是为了提高模型的表示能力和学习能力。通过增加网络的深度，可以增加网络的参数数量和非线性变换的层数，从而使网络能够学习更复杂、更抽象的特征和模式。

加深网络可以帮助模型更好地拟合目标函数，因为更深的网络可以学习到更多层次的特征表示，从而更好地捕捉输入数据的复杂性和抽象性。通过增加网络的深度，模型可以逐渐学习到更高级别的特征，使得模型能够更好地理解和表示输入数据中的结构和模式。

此外，加深网络还可以帮助模型找到更好的边界。更深的网络可以通过更多的非线性变换来建模输入数据的复杂关系，从而更好地分离不同类别之间的边界。通过增加网络的深度，模型可以更好地区分不同类别之间的样本，并更准确地进行分类或回归。

需要注意的是，加深网络并不总是能够带来性能的提升。在实践中，加深网络可能会遇到梯度消失、过拟合等问题。因此，在设计深度网络时，需要综合考虑网络的深度、宽度、正则化等因素，并进行适当的调整和优化，以获得更好的性能和泛化能力。

## 1.2 为什么线性层需要配合激活函数提高拟合能力

线性层本身只能完成线性变换，单独堆叠多个线性层仍然等价于一个线性层；真正提升非线性拟合能力的是“线性层 + 非线性激活函数”的组合。线性层的主要作用包括：

1. 特征组合：线性层可以对输入数据进行特征的线性组合。通过权重矩阵的乘法操作，线性层可以将输入特征按照一定的权重进行组合，从而生成更丰富的特征表示。这种特征组合能力使得模型能够捕捉到输入数据中的更多关联和非线性关系，提高了模型的表达能力。
2. 维度扩展：线性层可以将输入数据映射到更高维度的空间。通过增加特征的维度，线性层可以更好地拟合复杂的数据分布和决策边界。这种维度扩展的能力使得模型可以更好地捕捉到数据中的细微变化和模式，提高了模型的拟合能力。
3. 参数学习：线性层中的权重和偏置项是需要通过训练来学习的。通过反向传播算法和优化算法，模型可以自动学习到最优的权重和偏置值，以最小化损失函数。这样，线性层可以根据数据的特征和标签之间的关系进行参数调整，从而提高模型的拟合能力。

需要注意：如果两个线性层之间没有激活函数，那么它们的复合仍然是线性变换。因此，多层网络的表达能力主要来自线性层之间的非线性激活函数，以及足够的网络宽度和深度。

## 1.3 缩放点积注意力机制的作用

[注意力机制到底在做什么，Q/K/V怎么来的？一文读懂Attention注意力机制 (zhihu.com)](https://www.zhihu.com/tardis/zm/art/414084879?source_id=1005)

<img src="{{ '/assets/learning/27_deep_learning_foundations/image_001.png' | relative_url }}" alt="image-20260616134031229" style="zoom:50%;" />
$$
\begin{aligned}&\text{在这张图中,}Q\text{与}K^\top\text{ 经过MatMul,生成了相似度矩阵。对相似度矩阵每个元素除以 }\sqrt{d_k}\text{,}\\&d_k\text{为}K\text{ 的维度大小。这个除法被称为Scale。当}d_k\text{ 很大时,}QK^\top\text{ 的乘法结果方差变大,进行}\\&\text{Scale 可以控制注意力分数的尺度,训练时梯度更新更稳定。}\end{aligned}
$$
Mask 是机器翻译等自然语言处理任务中经常使用的环节。在机器翻译等 NLP 场景中，每个样本句子的长短不同，对于句子结束之后的位置，无需参与相似度的计算，否则影响Softmax 的计算结果。

将得到的归一化的权重矩阵与V词向量矩阵相乘。权重矩阵中某一行分别与词向量的一列相乘，词向量矩阵的一列其实代表着不同词的某一维度。经过这样一个矩阵相乘，相当于一个加权求和的过程，得到结果词向量是经过加权求和之后的新表示，而权重矩阵是经过相似度和归一化计算得到的。

## 1.4 Transformer的解码器中的两个多头注意力机制都是使用遮掩（mask）

- 填充掩码shape:(batch_size, max_seq_len)

- 注意力掩码shape: (batch_size, max_tar_len,max_tar_len)

 两者相加时，填充掩码会广播

- cross attention 掩码shape：(batch_size, max_tar_len,max_sour_len)


常见疑问：

(1) Transformer 的 Decoder 在一个 batch 中共享同一组模型参数。训练时不会在生成每个 token 后立刻更新权重，而是先并行计算整个目标序列的损失，再通过一次反向传播更新参数。自回归约束由 causal mask 保证：位置 t 只能看到位置 1 到 t 的历史 token，不能看到未来 token。

（2）训练时，每批次数据中decoder也会填补到和encoder填补的同样长度吗？

**在训练时的一个批次（batch）中，Decoder的输入序列需要填充到该批次中目标序列的最大长度，而Encoder的输入序列需要填充到该批次中源序列的最大长度。这两个最大长度 通常是不相等的。**

- **Encoder** 填充到 **源序列的最大长度** `L_src_max`。
- **Decoder** 填充到 **目标序列的最大长度** `L_tgt_max`。
- 这两个长度 `L_src_max` 和 `L_tgt_max` **通常不同**。
- 交叉注意力层能够处理这种长度差异，通过矩阵乘法 `Q (batch_size, L_tgt_max, d_model) * K^T (batch_size, d_model, L_src_max)` 产生注意力分数矩阵 `(batch_size, L_tgt_max, L_src_max)`。
- 填充掩码被用来在计算和最终的输出中忽略所有填充位置（无论是在Encoder还是Decoder侧）。


（3）在训练时，Encoder 与 Decoder 都有填充，那么在cross attention时应该使用哪个的填充掩码呢？

- Encoder输出：提供Key和Value矩阵，形状为`(batch_size, src_len, hidden_dim)`其中可能包含填充位置（如句末的`PAD`符号）。
- **Decoder输入**：提供Query矩阵，形状为 `(batch_size, tgt_len, hidden_dim)`，其填充掩码用于Decoder的自注意力层，但**不用于Cross-Attention**（因为在第一层MHA处理填充掩码后，则对应维度为0，后续则对计算不影响，如果为了减少计算量，则需要再次处理）。


（4）mask对应填充位置对应的注意力分数设为负无穷（-inf）

- 在 Decoder 的交叉注意力计算中，这个 mask 会被用来将 Encoder 输出序列中填充位置对应的注意力分数设为负无穷（-inf）。具体来说，当计算 Query (来自 Decoder) 和 Key (来自 Encoder 输出) 的注意力分数时，对于每个 Decoder 的位置，其注意力分数矩阵中对应 Encoder 填充位置的那些值会被设为 -inf。这样，在 softmax 后，对应维度位置会得到为0，Decoder 生成当前词元时，不会从 Encoder 的填充部分获取信息。
- 在seq2seq中，如果是在计算最终损失时用掩码矩阵计算，则需要将填充位置设为0。

**Encoder 自注意力中的填充掩码 (Padding Mask in Encoder Self-Attention):**

- **目的：** 在计算Encoder某个位置的词（例如，第 `i` 个词）对其所有位置词（包括它自己和其他词）的注意力分数时，我们只希望它关注那些**真实**的词，而**忽略**那些填充的词（PAD tokens）。
- **机制：** 填充掩码是一个布尔矩阵（或数值矩阵，如0和-∞），其形状为 `[batch_size, seq_len]`。

**为什么需要将填充掩码传递给Decoder？**

- **交叉注意力 (Cross-Attention):** Decoder在生成每个词时，会通过交叉注意力层来关注Encoder的输出KV表示。如果Encoder的输出表示中包含了填充位置的信息（即使自注意力时忽略了填充词，经过softmax为0，但是在经过与V相乘后填充位置仍然有向量表示），Decoder也应该忽略这些填充位置。因此，用于Encoder自注意力的**相同**填充掩码（或者基于Encoder输出位置生成的等效掩码）需要传递给Decoder的交叉注意力层，以屏蔽对Encoder填充位置的关注。


Transformer 的解码器中的两个多头注意力机制都是使用遮掩（mask）来限制模型在生成每个单词时只能依赖于之前已生成的单词。这是为了确保生成的单词不会依赖于后续的单词，从而保持解码器的自回归属性。在Decoder训练中，有两个遮掩操作：

- 因果遮掩（Causal Mask）：用在第一个Encoder MHA，在解码器的自注意力机制中，通过将未来位置的注意力分数设置为负无穷大，实现了对未来位置的遮掩。这样，解码器在生成每个单词时只能依赖于之前已生成的单词。

- 填充遮掩（Padding Mask）：

 - 在**批处理训练**时，Decoder的输入序列（即目标语言的标签序列）**通常也需要填充到相同长度**。（**所以Decoder 第一个 MHA 的 Mask 是既考虑到了attention mask，也考虑到了的填充mask**）

```
 1. Causal Mask 注意力掩码
 
 import torch
 seq_len = 5 # 序列长度
 causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
 # 输出：
 # tensor([[False, True, True, True, True],
 # [False, False, True, True, True],
 # [False, False, False, True, True],
 # [False, False, False, False, True],
 # [False, False, False, False, False ]])

 2. Padding Mask（填充掩码）
 屏蔽目标序列中的填充符号（如PAD），避免模型关注无效位置。
 实现代码：
 target_seq = torch.tensor([[1, 2, 3, 0, 0]]) # 假设0为PAD
 padding_mask = (target_seq 0)
 # shape: [batch_size, seq_len]
 # 输出一个seq的：tensor([[False, False, False, True, True]])

 3. 合并两种掩码
 通过逻辑或（OR）操作将两者合并，确保同时屏蔽未来位置和填充位置：

 combined_mask = causal_mask | padding_mask.unsqueeze(1)
 # 输出形状：[batch_size, 1, seq_len, seq_len]
 # 示例输出（batch_size=1）：
 # tensor([[[[False, True, True, True, True],
 # [False, False, True, True, True],
 # [False, False,False, True, True],
 # [False,False,False, True, True],
 # [False,False,False, True, True
 ]]]])
```

 > 个人觉得最终和上述得到的combined_mask掩码还是有问题，应该最后两行也要为Ture，否则就要在最后计算损失时需要根据decoder 输入target_seq的valid_len来重新计算损失！（与seq2seq一样）

 代码:

```python
def create_combined_mask(target_seq, pad_id):
    seq_len = target_seq.size(1)
    # 1. 生成Causal Mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(target_seq.device)
    # 2. 生成Padding Mask
    padding_mask = (target_seq pad_id) # [batch_size, seq_len]
    # 3. 合并掩码
    combined_mask = causal_mask | padding_mask.unsqueeze(1) # [batch_size, 1, seq_len, seq_len]
    return combined_mask

# 使用示例
target_seq = torch.tensor([[1, 2, 3, 0, 0]]) # 假设0为PAD
combined_mask = create_combined_mask(target_seq, pad_id=0)
```

 - Decoder的编码-解码器cross attention机制中，通过将填充位置的注意力权重设置为负无穷大，实现了对填充位置的遮掩。这样，解码器在生成每个单词时不会受到填充位置的干扰。

 在 Transformer 的Encoder-Decoder Cross-Attention中，由于Encoder可能存在填充（padding），此时使用的掩码是**Encoder的Padding Mask**，而不会使用Decoder的Padding Mask或Causal Mask。

 Decoder的Padding Mask和Causal Mask仅用于Decoder的自注意力层（Self-Attention），与Cross-Attention无关。


这两个遮掩操作确保了解码器在生成每个单词时只能依赖于前面已生成的单词，并且不会受到填充位置和未来位置的干扰。这样，Transformer 的解码器能够逐步生成输出序列，保持了自回归的性质。


seq2seq掩码不同：

decoder：会*核心理解*（teacher forcing），接收的encoder的最后一个时间步在隐藏层的隐藏状态

```python
 bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],evice=device).reshape(-1, 1)
……
 context = state[-1].repeat(X.shape[0], 1, 1)
 X_and_context = torch.cat((X, context), 2)
 output, state = self.rnn(X_and_context, state)
```

训练时：**会做填充遮掩**，不会做注意力遮掩（因为没有注意力softmax）

```python
 l = loss(Y_hat, Y, Y_valid_len)
```

预测时：使用具有预测最高可能性的词元，作为解码器在下一时间步的输入

```python
 for _ in range(num_steps):
 Y, dec_state = net.decoder(dec_X, dec_state) ### 【注意！！！】
 # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
 # Decoder的预测是self.dense = nn.Linear(num_hiddens, vocab_size)，
 # 所以最后输出为vocab_size列，所以与目标词典匹配
 dec_X = Y.argmax(dim=2)
```

transformer Decoder

decoder训练：在masked muti-head attention中输入都为x，且要做注意力遮掩，且训练时要做填充遮掩

```python
key_values = X
```

预测时：

```python
 key_values = torch.cat((state[2][self.i], X), axis=1) #prediction，存前面的输出
 '''block1 输入key_values: <bos> ;输出 <500>
 block1 输入key_values：<bos> <500> ;
 ……
 '''
 state[2][self.i] = key_values
```

在transformer中的decoder中，第二个多头注意力机制self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)中使用了encoder的enc_valid_lens掩码是为了避免decoder在生成输出时，参考了encoder中填充位置的信息。

在transformer中，encoder对输入序列进行编码，并生成了enc_outputs作为decoder的输入。由于输入序列的长度是不固定的，为了使得每个样本在经过encoder后的输出enc_outputs的长度一致，encoder会在输入序列的末尾添加一些填充符号。这些填充符号对于输入序列的实际含义并没有贡献，但是在计算注意力权重时，由于填充符号的存在，可能会导致注意力机制将注意力放在填充位置上，从而影响模型的性能。

为了避免这种情况发生，我们可以使用enc_valid_lens来对填充位置进行掩码操作。enc_valid_lens是一个长度为batch_size的向量，其中的每个元素表示对应样本的有效长度。在计算注意力权重时，我们可以将填充位置的enc_valid_lens设置为0，这样在计算注意力权重时，填充位置对应的权重就会被置为0，从而不会对模型的输出产生影响。

因此，在decoder的第二个多头注意力机制中，使用encoder的enc_valid_lens掩码是为了避免注意力机制将注意力放在填充位置上，从而提高模型的性能。

## 1.5 为什么L2范数可以防止过拟合

L2范数可以防止过拟合的原因如下：

1. 权重衰减：L2范数在损失函数中引入了权重衰减（weight decay）项，该项惩罚模型中较大的权重。通过惩罚较大的权重，L2范数可以使模型更加趋向于选择较小的权重，从而降低模型的复杂度。这有助于防止模型过度拟合训练数据，使模型更加泛化。

2. 特征选择：L2范数的权重衰减机制可以将一些特征的权重逐渐减小甚至为零，从而实现特征选择的效果。对于一些不重要的特征，它们的权重会被惩罚减小，这使得模型更加关注那些对预测目标有更大影响的重要特征。通过特征选择，L2范数可以减少不必要的特征对模型的影响，从而降低过拟合的风险。

3. 平滑性：L2范数可以使模型的权重分布更加平滑。较大的权重会被惩罚减小，而较小的权重则会受到保护。这种平滑性有助于减少模型对训练数据中的噪声和异常值的敏感性，提高模型的鲁棒性，从而减少过拟合的可能性。

综上所述，L2范数通过权重衰减、特征选择和平滑性等机制，可以有效地防止模型过拟合，提高模型的泛化能力。

## 1.6 缩放点积注意力中为何要除以 $\sqrt{d}$

这句话描述了在缩放点积注意力机制中，点积方差过大会对 **softmax函数**和模型训练产生的负面影响。以下是详细解释：

### 1. 点积方差过大的影响

在缩放点积注意力中，点积 $\mathbf{q} \cdot \mathbf{k}$ 的方差为 $ d $（假设查询和键的每个维度上的元素是独立随机变量，均值为 0，方差为 1）。如果维度 $ d $ 很大，点积的方差会线性增长，导致点积的值可能变得非常大（正或负）。

### 2. softmax 函数的性质

softmax 函数的定义如下：
${softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$

其中，$z_i$ 是输入值（在这里是点积结果），$ n $ 是输入向量的长度。

softmax 函数的特性是：

- 它对输入值的**相对大小**敏感。
- 如果某个 $ z_i $ 远大于其他 $ z_j $，则 ${softmax}(z_i) $会接近 1，而其他 ${softmax}(z_j)$ 会接近 0。

### 3. 点积方差过大的后果

如果点积的方差过大（即点积的值可能非常大），softmax 的输入值 $ z_i $ 也会变得非常大。这会导致以下问题：

#### （1）softmax 输出接近 one-hot 向量

- 当某个 $ z_i $ 远大于其他 $ z_j $ 时，$ e^{z_i} $ 会远大于其他 $ e^{z_j} $。
- 因此，softmax(z_i) 会接近 1，而其他 softmax(z_j)会接近 0。
- 这意味着 **softmax 的输出会接近一个 one-hot 向量**，即几乎所有的概率质量集中在某一个位置。

#### （2）梯度消失问题

- softmax 的梯度为：
 $\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \text{softmax}(z_i) (\delta_{ij} - \text{softmax}(z_j))$
  其中，$\delta_{ij}$ 是 Kronecker delta 函数（当 $ i = j $ 时为 1，否则为 0）。

- 如果 softmax 的输出接近 one-hot 向量（当只考虑这个情况时，对one-hot 向量softmax计算），即${softmax}(z_i) \approx 1 $且${softmax}(z_j) \approx 0$（对于 $j \neq i$），则梯度会变得非常小：
 $
  \frac{\partial \text{softmax}(z_i)}{\partial z_j} \approx 0$

- 这种现象称为**梯度消失**，会导致模型参数更新缓慢，甚至停止学习。

### 4. 为什么除以 $ \sqrt{d} $ 可以解决这个问题？

通过将点积结果除以 $\sqrt{d}$，可以将点积的方差从 $ d $ 归一化为 1：
$$
\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}}\right) = \frac{\text{Var}(\mathbf{q} \cdot \mathbf{k})}{d} = \frac{d}{d} = 1
$$
因为
$$
D(X) = E(X^2)-(EX)^2 = 1，\\
D(XY) = E(X^2Y^2) - (EXY)^2 = 1
$$
这样做的效果是：

- 点积的值被限制在一个合理的范围内，不会过大或过小。
- softmax 的输入值 $ z_i $ 也会被限制在一个合理的范围内，避免 **softmax 输出接近 one-hot 向量**。
- 梯度消失问题得到缓解，模型训练更加稳定。

---

### 5. 总结

- **点积方差过大**：导致 softmax 的输入值过大，使得 softmax 输出接近 one-hot 向量。
- **softmax 输出接近 one-hot 向量**：导致梯度消失问题，影响模型训练。
- **除以 $ \sqrt{d} $**：将点积的方差归一化为 1，限制 softmax 的输入值范围，避免梯度消失问题，使模型训练更加稳定。

这种设计是缩放点积注意力机制的核心，确保了注意力机制在高维空间中的有效性和稳定性。


## 1.7 正则化

**正则化（Regularization）** 是机器学习和深度学习中一种重要的技术，用于防止模型过拟合（Overfitting），从而提高模型的泛化能力（Generalization）。过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象。正则化通过在模型训练过程中引入额外的约束或惩罚，限制模型的复杂度，从而避免模型过度拟合训练数据。

正则化的核心思想是通过对模型施加某种约束，使模型在拟合训练数据的同时保持简单。简单的模型通常具有更好的泛化能力，能够更好地适应未见过的数据。

正则化的方法可以分为两类：

1. **显式正则化**：通过在损失函数中添加正则化项，直接限制模型的复杂度。
2. **隐式正则化**：通过修改模型结构或训练过程，间接限制模型的复杂度。

### 1. 常见的正则化方法

#### （1）权重衰减 L1 正则化（Lasso 正则化）

- 在损失函数中添加模型权重的 L1 范数（绝对值和）作为正则化项：
 $
  L_{\text{new}} = L_{\text{original}} + \lambda \sum_{i} |w_i|
  $
  其中：
 - $ L_{\text{original}} $ 是原始损失函数。
 - $ \lambda $ 是正则化系数，控制正则化的强度。
 - $ \sum_{i} |w_i| $ 是权重的 L1 范数。

- **作用**：
 - 倾向于产生稀疏权重，即许多权重被置为 0，从而实现特征选择。

#### （2）权重衰减 L2 正则化（Ridge 正则化）

- 在损失函数中添加模型权重的 L2 范数（平方和）作为正则化项：
 $
  L_{\text{new}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$
  其中：
 - $ \sum_{i} w_i^2 $ 是权重的 L2 范数。

- **作用**：
 - 限制权重的大小，防止权重过大，从而避免模型过拟合。

#### （3）Dropout

- 在训练过程中随机丢弃（置零）神经网络中的部分神经元。
- **作用**：
 - 防止神经元之间的共适应（co-adaptation），提高模型的泛化能力。

#### （4）早停（Early Stopping）

- 在训练过程中监控验证集的性能，当验证集性能不再提升时提前停止训练。
- **作用**：
 - 防止模型在训练集上过度拟合。

#### （5）数据增强（Data Augmentation）

- 通过对训练数据进行变换（如旋转、缩放、翻转等），生成更多的训练样本。
- **作用**：
 - 增加数据的多样性，提高模型的泛化能力。

#### （6）批量归一化（Batch Normalization）

- **在每一层的输入上进行归一化**，使其均值为 0，方差为 1。
- **作用**：
 - 加速训练，同时具有一定的正则化效果。

#### （7） 熵正则化

熵在信息论中是一个度量随机变量不确定性的指标，而在机器学习领域，熵正则化通常用于调整模型输出的概率分布，使其更加平滑或多样化。

熵正则化可以通过在损失函数中添加一个熵项来实现。这个熵项通常是与模型输出概率分布的熵成比例的，它可以鼓励模型探索更多的可能性，而不是过于自信地只选择某一个输出。

熵正则化也需要谨慎使用，因为过大的正则化系数可能会导致模型变得过于随机，影响其性能。因此，选择合适的正则化系数是应用熵正则化时的一个关键问题。

熵正则化的数学形式可以表示为：
$$
L=L_{original}−\lambda⋅H(p)
$$
其中：

- *L* 是正则化后的损失函数。

- $L_{original}$是原始的损失函数。

- $\lambda$是正则化系数，用于控制熵项的影响程度。

- *H*(*p*) 是模型输出概率分布 *p* 的熵，计算公式为 *H*(*p*)=−∑*p*(*x*)log*p*(*x*)。


对于一个离散随机变量 X*X*，其取值为 {*x*1,*x*2,…,xn}，对应的概率分布为 P(X=xi)=pi，熵 H*(*X) 的计算公式为：$H(X)=-\sum_{i=1}^{n}p_{i}\log p_{i}$

对于一个连续随机变量 X，其概率密度函数为 f(x)，熵 *H*(*X*) 的计算公式为：$$H(X) = - \int_{-\infty}^{\infty} f(x) \log f(x) dx$$

如果未引入熵正则化，模型的输出分布可能变得过于尖锐（集中在少数高概率词上），从而导致重复。

---

### 2. 正则化的作用

1. **防止过拟合**：

 - 通过限制模型的复杂度，避免模型过度拟合训练数据。

2. **提高泛化能力**：

 - 使模型在未见过的数据上表现更好。

3. **改善模型稳定性**：

 - 使模型的训练过程更加稳定，减少对噪声数据的敏感性。

---

### 4. 正则化的直观理解

可以将正则化理解为对模型的“惩罚”。模型在训练时不仅要最小化损失函数，还要尽量满足正则化项的约束。例如：

- L2 正则化会惩罚较大的权重，使权重趋向于较小的值。
- Dropout 会惩罚神经元的共适应，使网络更加鲁棒。

---

### 5. 正则化的数学形式

正则化通常通过在损失函数中添加正则化项来实现：
$
L_{\text{new}} = L_{\text{original}} + \lambda \cdot R(\mathbf{w})
$
其中：

- $ L_{\text{original}} $ 是原始损失函数。
- $ R(\mathbf{w}) $ 是正则化项，通常是权重的某种范数（如 L1 范数、L2 范数）。
- $ \lambda $ 是正则化系数，控制正则化的强度。

### 6. 总结

正则化是一种通过限制模型复杂度来防止过拟合的技术。常见的正则化方法包括 L1 正则化、L2 正则化、Dropout、早停、数据增强等。正则化的核心思想是使模型在拟合训练数据的同时保持简单，从而提高模型的泛化能力。


## 1.8 何处使用mask

- 算注意力时，因为用了padding，所以需要用mask。
- 在计算损失时，需要用到mask，padding部分不需要计算损失。具体可以先把包含padding部分的损失一起算出并且不用reduction降维(self.reduction='none')，最后再通过mask矩阵去除掉padding部分的损失。（在61序列到序列章节）

## 1.9 自注意力机制为什么是并行的

**自注意力机制（Self-Attention Mechanism）** 之所以是并行的，主要是因为它的计算过程可以分解为独立的矩阵运算，这些矩阵运算可以高效地并行化处理。以下是详细解释：

---

### 1. 自注意力机制的核心计算步骤

<img src="{{ '/assets/learning/27_deep_learning_foundations/figure_010.png' | relative_url }}" alt="image-20250121163303739" style="zoom:80%;" />

注意力机制最初用于序列到序列（Seq2Seq）模型（如机器翻译），用于在解码过程中动态地关注编码器的不同部分。注意力机制用于建模两个序列之间的关系。Key 和 Value 来自编码器的输入序列，而 Query 来自解码器的输入序列。

**注意力机制应用场景：**

- **Seq2Seq 模型**：在解码过程中，解码器通过注意力机制动态地关注编码器的不同部分。
- **图像描述生成**：在生成图像描述时，模型通过注意力机制关注图像的不同区域。
- **语音识别**：在语音识别中，模型通过注意力机制关注语音信号的不同时间段。

- 自注意力机制是 Transformer 的核心组件，用于建模输入序列中不同元素之间的关系。

自注意力机制的核心是通过**计算输入序列中每个元素与其他元素的相关性**（注意力权重），然后根据这些权重对输入序列进行加权求和。具体步骤如下：

**自注意力机制应用场景：**

- **Transformer 模型**：自注意力机制是 Transformer 的核心组件，用于建模输入序列中不同元素之间的关系。
- **文本分类**：在文本分类任务中，自注意力机制可以捕捉文本中不同词之间的关系。
- **图像处理**：在图像处理任务中，自注意力机制可以捕捉图像中不同区域之间的关系。

#### （1）计算查询（Query）、键（Key）和值（Value）

- 输入序列 $ \mathbf{X} \in \mathbb{R}^{n \times d} $（$ n $ 是序列长度，$ d $ 是特征维度）。
- 通过线性变换得到查询 $ \mathbf{Q} $、键 $ \mathbf{K} $ 和值 $ \mathbf{V} $：
 $
  \mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V
  $
  其中，$ \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V $ 是可学习的权重矩阵。

#### （2）计算注意力分数

- 通过点积计算查询和键的相似度（注意力分数）：
 $
  \mathbf{A} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}
  $
  其中，$ \mathbf{A} \in \mathbb{R}^{n \times n} $ 是注意力分数矩阵。

#### （3）计算注意力权重

- 对注意力分数矩阵 $ \mathbf{A} $ 进行 softmax 操作，得到注意力权重矩阵：
 $
  \mathbf{P} = \text{softmax}(\mathbf{A})
  $
  其中，$ \mathbf{P} \in \mathbb{R}^{n \times n} $ 是注意力权重矩阵。

#### （4）**加权求和

- 使用注意力权重对值 $ \mathbf{V} $ 进行加权求和，得到输出序列：
 $
  \mathbf{O} = \mathbf{P} \mathbf{V}
  $
  其中，$ \mathbf{O} \in \mathbb{R}^{n \times d} $ 是输出序列。

---

### 2. 为什么自注意力机制是并行的？

自注意力机制的计算过程可以分解为多个独立的矩阵运算，这些矩阵运算可以高效地并行化处理。以下是具体原因：

#### （1）矩阵乘法的并行性

- 计算 $ \mathbf{Q} = \mathbf{X} \mathbf{W}_Q $、$ \mathbf{K} = \mathbf{X} \mathbf{W}_K $、$ \mathbf{V} = \mathbf{X} \mathbf{W}_V $ 都是矩阵乘法，矩阵乘法本身可以高度并行化。
- 现代硬件（如 GPU 和 TPU）对矩阵乘法有高度优化的并行实现。

#### （2）注意力分数的并行计算

- 计算 $ \mathbf{A} = \mathbf{Q} \mathbf{K}^T $ 也是一个矩阵乘法，可以并行化。
- softmax 操作可以逐行并行计算。

#### （3）加权求和的并行性

- 计算 $ \mathbf{O} = \mathbf{P} \mathbf{V} $ 是矩阵乘法，可以并行化。

#### （4）序列长度的独立性

- 自注意力机制对序列中每个元素的处理是独立的，因此可以并行计算每个元素的注意力权重和输出。

---

### 3. 与 RNN 的对比

- **RNN（循环神经网络）**：
 - RNN 的计算是顺序的，必须按时间步依次计算，无法并行化。
 - 长序列训练和推理速度较慢。
- **自注意力机制**：
 - 自注意力机制的计算是并行的，可以同时处理整个序列。
 - 训练和推理速度更快，尤其适合长序列任务。

---

### 4. 并行化的硬件支持

- **GPU/TPU**：
 - 现代深度学习硬件（如 GPU 和 TPU）对矩阵运算有高度优化的并行支持。
 - 自注意力机制的计算过程主要由矩阵乘法组成，因此可以充分利用硬件并行性。

---

### 5. 总结

自注意力机制之所以是并行的，是因为它的核心计算步骤（如矩阵乘法、softmax、加权求和）可以分解为独立的矩阵运算，这些运算可以高效地并行化处理。与 RNN 的顺序计算不同，自注意力机制可以同时处理整个序列，从而显著提高计算效率，尤其适合长序列任务。这种并行性是自注意力机制在 Transformer 等模型中广泛应用的重要原因之一。

## 1.10 **Transformer** 的**Encoder** 和 **Decoder** 

在 **Transformer** 模型中，**Encoder** 和 **Decoder** 的设计目标不同，因此它们的处理数据方式也有所不同：

- **Encoder 是双向的**：可以同时看到输入序列的全部信息。
- **Decoder 是单向的**：只能看到当前时刻及之前的信息。

这种设计是为了满足不同任务的需求，以下是详细解释：

------

### 1. Encoder 的双向性

#### 设计目标

Encoder 的任务是对输入序列进行编码，提取出序列中每个元素的特征表示。为了充分捕捉序列中每个元素与上下文的关系，Encoder 需要能够同时看到整个输入序列。

#### 实现方式

- **自注意力机制（Self-Attention）**：
 - Encoder 使用自注意力机制计算每个元素与序列中所有其他元素的关系。
 - 在计算注意力权重时，每个元素都可以“看到”整个序列的信息。
 - 因此，Encoder 是双向的。

#### 示例

假设输入序列是 `[A, B, C, D]`，Encoder 在计算每个元素的表示时：

- 对于元素 `B`，Encoder 可以同时看到 `A`、`C` 和 `D` 的信息（计算query和key的点积时，能够看到）。
- 这种双向性使得 Encoder 能够捕捉到序列中每个元素的全局上下文信息。

#### 应用场景

- **文本编码**：在机器翻译任务中，Encoder 对源语言句子进行编码，需要捕捉句子中每个词的上下文信息。
- **图像编码**：在图像处理任务中，Encoder 对图像的特征进行编码，需要捕捉图像中每个区域与全局的关系。

------

### 2. Decoder 的单向性

#### 设计目标

Decoder 的任务是生成输出序列（如翻译结果、文本生成等）。在生成序列时，Decoder 需要遵循序列生成的顺序，即只能基于已经生成的部分来预测下一个元素。

#### 实现方式

- **掩码自注意力机制（Masked Self-Attention）**：
 - Decoder 使用掩码自注意力机制，确保在计算注意力权重时，每个元素只能“看到”当前时刻及之前的信息。
 - 具体来说，Decoder 在计算注意力权重时，会使用一个上三角掩码矩阵（Upper Triangular Mask），将未来时刻的信息屏蔽掉。
 - 因此，Decoder 是单向的。

#### 示例

假设 Decoder 正在生成序列 `[X, Y, Z]`：

- 在生成 `Y` 时，Decoder 只能看到 `X` 和 `Y` 的信息，而不能看到 `Z` 的信息。
- 这种单向性确保了 Decoder 在生成序列时不会“偷看”未来的信息。

## 1.11 自注意力机制中 K、V、Q来自同一序列的目的

### （1）捕捉序列内部的关系

- 自注意力机制的核心目标是建模序列中不同元素之间的关系。
- 通过让 **K**、**V**、**Q** 都来自同一个序列，模型可以计算序列中每个元素与其他元素的相似度（注意力权重），从而捕捉序列内部的依赖关系。

#### （2）动态权重分配

- 自注意力机制通过计算 Q**Q** 和 K**K** 的点积，得到注意力权重，然后对 V**V** 进行加权求和。
- 这种动态权重分配机制使得模型可以根据输入序列的内容，自适应地关注序列中不同部分的信息。

#### （3）并行计算

- 由于 K**K**、V**V**、Q**Q** 都来自同一个序列，计算过程可以高效地并行化，从而提高计算效率。

#### （4）统一表示

- 通过将 K**K**、V**V**、Q**Q** 都来自同一个序列，模型可以将输入序列的表示统一到一个共享的特征空间中，从而简化模型设计。


## 1.12 Transformer 为什么要使用多头注意力（Multi-Head Attention）

多头注意力是 Transformer 的核心组件之一，它的主要目的是增强模型的表达能力。以下是多头注意力的设计动机和优势：

### （1）捕捉不同的注意力模式

- 单一的注意力机制可能只能捕捉到一种类型的依赖关系（如局部依赖或全局依赖）。
- 多头注意力通过并行计算多个注意力头（Attention Head），每个头可以学习到不同的注意力模式，从而捕捉到更丰富的依赖关系。

#### （2）**增强模型的表达能力

- 每个注意力头可以关注序列中不同的子空间（Subspace），从而捕捉到不同的特征。
- 通过将多个注意力头的输出拼接起来，模型可以综合多种特征表示，增强表达能力。

#### （3）提高泛化能力

- 多头注意力机制可以看作是一种集成学习（Ensemble Learning），通过多个注意力头的组合，提高模型的泛化能力。

#### （4）并行计算

- 多头注意力机制可以并行计算多个注意力头，从而充分利用硬件（如 GPU 或 TPU）的并行计算能力，提高计算效率。


## 1.13 进入多头注意力的Q,K,V是否拆分

---

### 1. 多头注意力的输入

- 原始的 $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 是通过输入序列 $ \mathbf{X} $ 进行线性变换得到的：
 $
  \mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V
  $
  其中，$ \mathbf{W}^Q $、$ \mathbf{W}^K $、$ \mathbf{W}^V $ 是可学习的权重矩阵。

---

### 2. 多头注意力的实现

假设有 $ n $ 个注意力头，多头注意力的实现步骤如下：

#### （1）线性变换

- 对原始的 $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 分别进行 $ n $ 次线性变换，得到 $ n $ 组不同的 $ \mathbf{Q}_i $、$ \mathbf{K}_i $、$ \mathbf{V}_i $：
 $
  \mathbf{Q}_i = \mathbf{Q} \mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{K} \mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{V} \mathbf{W}_i^V
  $
  其中，$ \mathbf{W}_i^Q $、$ \mathbf{W}_i^K $、$ \mathbf{W}_i^V $ 是可学习的权重矩阵，$ i = 1, 2, \dots, n $。

- 这些线性变换的目的是将原始的 $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 投影到不同的子空间（Subspace），从而使每个注意力头可以关注不同的特征。

#### （2）计算注意力

- 对每组 $ \mathbf{Q}_i $、$ \mathbf{K}_i $、$ \mathbf{V}_i $ 分别计算注意力：
 $
  \text{Attention}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}}\right) \mathbf{V}_i
  $
  其中，$ d_k $ 是 $ \mathbf{K}_i $ 的维度。

#### （3）拼接和线性变换

- 将 $ n $ 个注意力头的输出拼接起来，然后进行线性变换：
 $
  \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{Attention}_1, \text{Attention}_2, \dots, \text{Attention}_n) \mathbf{W}^O
  $
  其中，$ \mathbf{W}^O $ 是可学习的权重矩阵。

---

### 3. 为什么使用不同的 $ \mathbf{Q}_i $、$ \mathbf{K}_i $、$ \mathbf{V}_i $

- **捕捉不同的特征**：通过将 $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 投影到不同的子空间，每个注意力头可以关注序列中不同的特征（如局部依赖、全局依赖等）。
- **增强表达能力**：不同的 $ \mathbf{Q}_i $、$ \mathbf{K}_i $、$ \mathbf{V}_i $ 使得模型可以捕捉到更丰富的依赖关系，从而提高模型的表达能力。
- **并行计算**：每个注意力头可以独立计算，从而充分利用硬件的并行计算能力。

---

### 4. 与“分割”的区别

- **分割**：如果将原始的 $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 直接分割为 $ n $ 个部分，每个部分的维度会减小，但每个头的输入仍然是相同的（只是维度不同）。
- **线性变换**：通过线性变换，每个头的输入是不同的，且可以捕捉到不同的特征。

## 1.14 注意力每个步骤的作用

计算注意力的每个步骤及其作用如下：

1. **计算点积 $QK^T$**：计算查询和键的相似度。
2. **缩放点积$\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}$**：避免点积结果过大，导致梯度消失。
3. **计算 softmax**：将注意力分数转换为概率分布。
4. **加权求和$\mathbf{P} \mathbf{V}$**：生成每个查询对应的输出表示。

这些步骤共同作用，使得注意力机制能够动态地关注输入序列中不同部分的信息，从而捕捉到序列中的依赖关系。

**加权求和 PV用于生成每个查询对应的输出表示是什么意思？**

- **查询 (Q)**：表示需要关注的内容。
- **键 (K)**：表示被关注的内容。
- **值 (V)**：表示被关注内容的具体信息。
- **注意力权重 (P)**：表示查询对每个键的关注程度。

加权求和 **PV** 的作用是将值矩阵 **V** 中的信息按照注意力权重 **P** 进行加权组合，从而生成每个查询对应的输出表示。

P是计算查询每个token在每个键上对应的token上的注意力权重，则PV表示将查询每个token在每个键上对应的token上的注意力分数与每个V上的token在每个维度下的表示(一个维度下的表示是一个数值，一维)加和，从而得到原来的query，经过Key和Value后的新表示，每个token在每一个维度下的数值都是经过所有token在当前维度上与注意力权重相乘后得到的数值，其他维度也是一样，从而学习到序列内部的每个token在与其他token之间的关注程度，这是在每个维度下实现的。

<img src="{{ '/assets/learning/27_deep_learning_foundations/figure_011.png' | relative_url }}" alt="image-20230904161353534" style="zoom:50%;" />


self-attention是用文本中的其它词来增强目标词的语义表示，但是目标词本身的语义还是会占主要部分的

## 1.15 特征空间

**特征空间**：

- **定义**：特征空间是指数据的所有特征（或属性）构成的空间。例如，如果一个数据点有 $ d $ 个特征，那么它所在的特征空间就是 $ \mathbb{R}^d $。
- **示例**：对于一个二维数据点 $ (x_1, x_2) $，其特征空间是二维平面 $ \mathbb{R}^2 $。

 **特征子空间（Feature Subspace）**

- **定义**：特征子空间是特征空间的一个子集，通常是通过对原始特征进行线性或非线性变换得到的低维空间。
- **特点**：
 - 特征子空间的维度通常比原始特征空间低。
 - 特征子空间捕捉了数据中某些特定的模式或结构，例如主要的变化方向或类别之间的差异。
 - 特征子空间通常捕捉了数据中最重要的变化方向或模式。例如，PCA 通过找到数据方差最大的方向，将数据投影到这些方向上，从而保留最重要的信息。

多头注意力机制上的投影：

- 在多头注意力机制中，**每个注意力头使用不同的线性变换矩阵**，将输入序列投影到不同的特征子空间。
- 尽管输入相同，并且投影后的维度可能相同，但由于变换矩阵不同（为了学习整个特征空间的某些特定的模式或结构，这对应着不同的特征子空间），这些特征子空间仍然是不同的，捕捉了数据中不同的特征。

## 1.16 Transformer的位置编码局限性

**固定位置编码**的局限性：标准的正弦-余弦位置编码是固定的，不可学习。它⽆法适应特定任务或数据的变化，因此在某些特定应⽤中，可能⽆法提供最优的位置信息。

**有限的序列长度**：正弦-余弦位置编码对序列长度有⼀定限制。在处理⾮常⻓的序列时，固定位置编码可能⽆法充分表达序列的全貌。

- 由于周期性，难以捕获长序列关系。 正余弦编码的波长呈几何级数增长（从 $2\pi$ 到 $10000\cdot 2\pi$），高频维度（短波长）对短距离位置敏感，低频维度（长波长）对长距离位置敏感。**长序列表现：**
  - 当序列长度超过最大波长（如 $L>10000$）时，低频维度的位置编码会重复振荡（即 $pos/10000^{2i/d}\ \bmod\ 2\pi$），导致不同位置编码相似。
  - 例如，位置 $pos=10000$ 和 $pos=20000$ 在低频维度的编码值几乎相同，模型无法区分。

- 频率问题，长序列的高位置中难以区分临近密集的位置编码：位置编码中的频率是由 $\frac{pos}{10000^{2i/d}}$决定的。随着位置 *pos* 的增加，频率会变小。**低频信号可能导致模型在学习和解码位置信息时出现困难，因为模型需要区分越来越密集的编码值。（这个时候一般是需要外推）**

**位置编码的线性组合**：**位置编码与词嵌⼊的线性组合可能导致信息丢失，因为位置编码的信息可能被嵌⼊中的信息所覆盖**， 从⽽影响模型对位置信息的敏感性。


优势：

- 捕捉序列顺序**（位置编码的本质）**：位置编码使得 Transformer 能够区分词语在序列中的位置，从而理解上下文信息。这对于处理自然语言序列尤其重要，因为序列的顺序对语义理解具有关键作用。

- 隐式相对位置：因为三角函数的平移性质，位置编码含有相对位置信息（因为周期原因，这种相对位置层现周期性）：

 **正弦-余弦编码：**

对每个位置 $t$ 和维度 $i$，我们通常会这样生成位置编码：

$$
\mathrm{PE}_{(t,2i)}=\sin\left(\frac{t}{10000^{2i/d}}\right),
\qquad
\mathrm{PE}_{(t,2i+1)}=\cos\left(\frac{t}{10000^{2i/d}}\right)
$$

简单来说，就是不同频率的正弦余弦波叠加而成。

**关键：三角函数的平移性质**

三角函数的一个数学特性：

$$
\sin(t+\Delta t)=\sin t\cdot \cos\Delta t+\cos t\cdot \sin\Delta t
$$

$$
\cos(t+\Delta t)=\cos t\cdot \cos\Delta t-\sin t\cdot \sin\Delta t
$$

也可以写成矩阵形式（非常像旋转矩阵）：

$$
\begin{pmatrix}
\sin(t+\Delta t) \\
\cos(t+\Delta t)
\end{pmatrix}
=
\begin{pmatrix}
\cos\Delta t & \sin\Delta t \\
-\sin\Delta t & \cos\Delta t
\end{pmatrix}
\begin{pmatrix}
\sin t \\
\cos t
\end{pmatrix}
$$

这就意味着什么呢？

**某个位置的编码经过一个“旋转矩阵”，就可以转化为另一个相对位置的编码。** 这正是相对位置信息的体现！也就是说，我们可以用某种固定的变换（旋转）来表达“当前位置 + 相对偏移”，这是一种 **隐式表达相对位置** 的能力！


为了克服这些限制，研究者们提出了多种改进方法：

- **可学习位置编码**：允许模型在学习过程中调整位置编码，以适应不同长度的序列。
- **相对位置编码**：关注位置之间的相对差异，而不是绝对位置，有助于处理更长序列。（在计算attention时依据**query和key的相对位置**差来添加偏置，使模型能够更好地处理局部依赖和长距离信
 息。）
- **改进的周期函数**：使用非周期性或更长周期的函数来生成位置编码。


## 1.17 LLM长文本分段

LLM（Large Language Model）在处理长序列文本时，通常会将这个很长的文本序列分成几段，每段长度相同，并且实现KV Cache。

具体来说，由于 LLM 的注意力机制（Attention Mechanism）在计算时需要考虑序列中的所有元素，当序列长度变得非常长时，计算复杂度会显著增加。为了解决这个问题，LLM 通常会将长序列分成多个较短的段，每个段的长度相同。这样，在计算注意力时，只需要考虑当前段内的元素，从而降低了计算复杂度。

同时，LLM 还会实现KV Cache（Key-Value Cache）来进一步优化计算。KV Cache是一种缓存机制，用于存储已经计算过的注意力分数和值向量。当模型在处理新的序列段时，可以重用这些缓存的数据，从而避免重复计算，提高效率。


## 1.18 学习率和梯度裁剪

学习率和梯度裁剪都是影响梯度的大小，那有什么区别？

- **调整时机**：学习率是在梯度计算之前就确定的，而梯度裁剪是在梯度计算之后进行的。
- **调整方式**：学习率是通过乘法调整梯度的大小，而梯度裁剪是通过限制梯度的最大范数（超参数）来调整梯度的大小，如果超过最大范数值，则按比例缩小（例如：缩放因子 = 最大允许范数 / 当前梯度范数）。
- **目的不同**：学习率主要用于控制收敛速度和稳定性，而梯度裁剪主要用于防止梯度爆炸，保证模型训练的稳定性。

## 1.19 输入到输出的映射关系

有监督学习与无监督学习是在学习输入到输出的映射关系，但与监督学习（如多分类深度学习）的不同之处在于，无监督学习没有明确的输出标签。无监督学习的目标是发现数据中的结构、模式或关系。

多分类深度学习的核心是学习一个复杂的非线性映射函数，该函数能够从高维输入空间映射到类别空间。

## 1.20 Logits

模型输出的原始logits。**Logits**（逻辑值，读作 /ˈloʊdʒɪts/）是神经网络在分类任务（如文本生成、图像分类）中最后一层线性层的原始输出值，**尚未经过Softmax归一化**概率分布。它们是模型对每个可能类别的“未校准”的置信度分数，可以是任意实数（正数、负数或零）。


## 1.21 负对数似然（Negative Log-Likelihood, NLL）损失函数和多分类交叉熵（Cross-Entropy）

负对数似然损失（NLL）要求输入通常是对数概率，真实类别为 $y$ 时：

$$
\mathrm{NLL}(y,p)=-\log p_y
$$

多分类交叉熵可写为：

$$
\mathrm{CE}(y,p)=-\sum_{i=1}^{C}y_i\log p_i
$$

其中 $y_i$ 为 one-hot 标签，$p_i=\mathrm{softmax}(z)_i$。在 PyTorch 中，`CrossEntropyLoss` 通常等价于 `LogSoftmax + NLLLoss`，输入应为未经过 Softmax 的 logits。

- **NLL Loss**：更底层，需显式提供对数概率，适合灵活定制。
- **Cross-Entropy Loss**：更高效，内置Softmax，适合标准分类任务。
- **核心关系**：Cross-Entropy = Softmax + NLL。


## 1.22 bert中tokenizer.json中有“##”

tokenizer.json中记录分词与token ID的映射：

```
"contribute":9002,"indonesian":9003,"pet":9004,"creates":9005,"##com":9006,
```

vocab.txt词汇表中记录分词：

```
52
defence
internal
persian
43
methods
##ging
arrested
```

在 BERT 的 `tokenizer.json` 文件中，`"##ffey": 24513` 这样的条目表示一个 **子词（subword）token**，其中前缀 `##` 是 BERT 的 **WordPiece 分词器** 的特殊标记，用于指示该 token 是一个 **非首字母开头的连续子词**（即前一个 token 和当前 token 应连续拼接，不能独立成词）。

### `##` 的作用

- 表示子词的延续：

 BERT 的 WordPiece 分词器会将罕见词拆分为更小的子词单元。

```
 ##
```

 前缀表示当前 token 必须与前一个 token 拼接，形成完整的词或词根。

 - 例如：`"unaffey"` 可能被拆分为 `["un", "##aff", "##ey"]`，其中 `"un"` 是独立子词，而 `"##aff"` 和 `"##ey"` 必须与前面的子词连接。

- **避免歧义**：
 没有 `##` 前缀的 token 可以独立出现（如 `"un"`），而有 `##` 前缀的 token 必须依附于前一个 token（如 `"##aff"` 不能单独使用）。

在 BERT 的 `tokenizer.json` 中，**`"ffey##"` 这样的带 `##` 后缀的 token 是** **不符合 WordPiece 分词器设计规范的**，通常不会出现在标准 BERT 的词汇表中。`##` 前缀的作用是明确标记子词的延续性（即必须与前一个 token 拼接）


## 1.23 Post-Norm VS Pre-Norm

论文：On Layer Normalization in the Transformer Architecture https://arxiv.org/pdf/2002.04745.pdf

<img src="{{ '/assets/learning/27_deep_learning_foundations/figure_012.png' | relative_url }}" alt="image-20251009151636109" style="zoom:50%;" />

[Pre-Norm vs. Post-Norm - 知乎](https://zhuanlan.zhihu.com/p/674704060)

Transformer 原始结构中，使用的是 post-norm，但是这种方式有收敛速度慢、参数难调的问题，pre-norm 是为了解决这些问题做出的改良尝试。

Pre-norm: $$ X_{t+1}=X_{t}+F_{t}(\text{Norm}(X_{t})) $$ (1)

Post-norm: $$ X_{t+1}=\text{Norm}(X_{t}+F_{t}(x_{t})) $$ (2)

### 1. 从**梯度传播**角度分析二者学习的难易程度

残差连接（Residual Connection，或称跳跃连接、Skip Connection）的核心目的是**解决深层神经网络训练中的梯度消失/爆炸问题，并提升模型性能**，其具体目的或作用：

### 2. 缓解梯度消失/爆炸，实现深层网络训练

- **问题背景**：在**深层网络**中，梯度通过多层反向传播时容易逐层衰减（梯度消失）或指数级放大（梯度爆炸）（梯度累乘则会爆炸（当每一个梯度大于1）或者消失为0（当每一个梯度接近0）），导致模型难以训练。
- **残差连接的作用**：通过将输入直接跳过部分层（即 *F*(*x*)+*x* 的结构），这样在反向传播中，**部分梯度可以绕过非线性变换层**，即使**中间层的梯度很小**，**残差项**也能直接梯度回传，保证梯度不会完全消失，**从而支持更深的网络训练**。

#### Post-norm

令某个维度的 $x$ 的方差为 $\sigma_1^2$ 而 $F(x)$ 的方差为 $\sigma_2^2$，并且假设两者相互独立（这个假设有点强，但不影响对下述分析的直观理解），那么 $x+F(x)$ 的方差为 $\sigma_1^2+\sigma_2^2$。假设初始状态下 $x,F(x)$ 的方差均为 1（注意：常见的参数初始化目标就是使得 $F(x)$ 的方差和 $x$ 的方差一致），那么 $x+F(x)$ 的方差就是 2，而 Post Norm（L2）操作会将方差重新降为 1，这相当于

$$
x_{t+1}=\frac{x_t+F_t(x_t)}{\sqrt{2}}
$$

递归下去可得

$$
\begin{aligned}
x_l
&=\frac{x_{l-1}}{\sqrt{2}}+\frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
&=\frac{x_{l-2}}{2}+\frac{F_{l-2}(x_{l-2})}{2}+\frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
&=\cdots \\
&=\frac{x_0}{2^{l/2}}+\frac{F_0(x_0)}{2^{l/2}}+\frac{F_1(x_1)}{2^{(l-1)/2}}+\frac{F_2(x_2)}{2^{(l-2)/2}}+\cdots+\frac{F_{l-1}(x_{l-1})}{2^{1/2}}
\end{aligned}
\tag{6}
$$

其中 $x_0$ 表示 embedding 层的输入。根据上面的公式我们可以大致猜测，$\frac{\partial x_l}{\partial x_i}$ 是呈 $l-i$ 的指数衰减的，[1] 根据 $F_l(x)$ 的具体形式推出了 $\frac{\partial x_l}{\partial x_i}$ 的大小，验证了我们的猜测。对于 RMS Norm，我们可以进行类似的分析，此时我们可以假设 $x,F(x)$ 的二阶矩为 1 且独立，由此也能得到上面类似的推导。

由此可以看出，Post-Norm 会导致梯度的指数衰减，这其实是和残差连接引入 highway-connection 的思想相悖的，从而导致模型比较难学。另外 [4] 的分析表明，在初始化不好、没有 warm-up 的情况下，底层的 $\sigma_1,\sigma_2,\cdots$ 的增长速度很快（系数 2 会变得更大），也即梯度衰减会更厉害，导致模型难以学习。



这个结构跟pre-norm比起来充分考虑了所有分支（残差）的输出，做到了真正增加深度，自然精度会相对好一些。

不过它也有它很显然的问题，当t足够大、也就是叠加的attention层足够多以后，底层那些分支（残差）的影响力被衰减掉了，那么势必会有梯度消失的问题，这也就是文章开头所说的post-norm难收敛、参数难调的原因。本来我们做Norm也是为了处理梯度消失，但从分析看来，transformer结构中的[layernorm](https://zhida.zhihu.com/search?content_id=235416194&content_type=Article&match_order=1&q=layernorm&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTQ0NDM5MjEsInEiOiJsYXllcm5vcm0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzU0MTYxOTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.GKWb7wK2pmW3iHFgsRFwIPRvwpbDJqU-CHkhPIP79C4&zhida_source=entity)并没有完全实现它的作用。那这就意味着transformer原始结构的失败吗？并不是的，因为这种梯度消失的问题在整个结构上来看（配合上adam系优化器和学习率warmup）是并不明显的。

#### Pre-norm：

相对的，在使用 Pre-Norm 的情况下迭代展开之后可以认为初始阶段有：

$$
x_1=x_0+F_0(x_0),
$$

$$
x_2=x_1+F_1\left(\frac{x_1}{\sigma_1}\right)
=x_0+F_0(x_0)+F_1\left(\frac{x_1}{\sqrt{2}}\right),
\tag{7}
$$
$$
\vdots
$$

$$
x_l=x_0+F_0(x_0)+F_1(x_1/\sqrt{2})+F_2(x_2/\sqrt{3})+\cdots+F_{l-1}(x_{l-1}/\sqrt{l})
$$

其中 $\frac{x_i}{\sqrt{i}}$ 都是均值为 0，方差为 1 的变量。可以看出，Pre-Norm 形式下，$x_l,x_0$ 之间有个等式通道，使得梯度能更快速无衰减地从 $x_l$ 传递到 $x_0$，使得模型更好优化。**当然，这样最后的 $x_l$ 方差将会很大，所以在接预测层之前 $x_l$ 也还要加个 Normalization。**



总结：

- Post-Norm中，当前层参数梯度对应的**残差项**随着参数所在层的增加层现指数衰减（ ${2/3}^l$），层数越深，**前面的层数对应的参数的梯度随着层数的加深呈现指数衰减**（第一层：${2/3}^l$，……，第i层${2/3}^{l-i}$），则会造成梯度消失）
- Pre-Norm中，当前层参数梯度几乎不随层数的加深而变化

总结来说，**Pre-Norm的残差连接效应强，使得梯度的回传更容易**，而Post-Norm的应用会使得梯度快速衰减，导致模型难以学习。

### 3. 从**模型深度**比较二者学习效果的好坏

**直观解释：** 残差链接会导致模型实际深度小于模型的层数，这个本来也是残差连接的特点。根据上面的分析，Pre-Norm 是典型的全局残差连接，而 Post-Norm 因为 Layer-Norm 的引入改变了这种残差连接，使得在相同层数的情况下，Post-Norm 的实际深度比 Pre-Norm 更深，所以效果更好。

唐翔是从更为理论的角度进行了分析：

$$
x_l=x_0+F_0(x_0)+F_1(x_1/\sigma_1)+\cdots+F_{l-1}(x_{l-1}/\sigma_{l-1}),
$$

$$
x_{l+1}=x_0+F_0(x_0)+F_1(x_1/\sigma_1)+\cdots+F_{l-1}(x_{l-1}/\sigma_{l-1})+F_l(x_l/\sigma_l)
$$

在层数 $l$ 较大时，因为 normalization 的原因，$F_l(x_l/\sigma_l)\ll x_l$，所以 $x_{l+1}\approx x_l$，此时

$$
x_{l+2}=x_l+F_l(x_l/\sigma_l)+F_{l+1}(x_{l+1}/\sigma_{l+1})
\approx x_l+F_l(x_l/\sigma_l)+F_{l+1}(x_l/\sigma_l)
\tag{8}
$$

**也即在层数较大部分模型实际上更像扩展了宽度，而不是深度，所以相对好训练。但一般而言，同参数量下，扩展深度比扩展宽度效果更好。** 相较而言，Post-Norm 的每一层输出都做了 normalization，令 $o_{l-1}$ 表示第 $l-1$ 层做了 normalization 之后的输出，则第 $l$ 层引入的增量 $F(o_{l-1})$ 相较于 $o_{l-1}$ 仍然较大（这一点从公式 6 也可以看出）。所以在同参数量下，Post-Norm 表现比 Pre-Norm 更好。

根据上面的分析，可以得出结论，在**同等层数和参数量**的情况下，Post-Norm的实际深度比Pre-Norm更深，导致其在训练充分的情况下比Pre-Norm效果更好。但是Post-Norm缺乏全局的残差连接，因为**Layer-Normalization的引入有导致梯度消失的风险**，导致其训练上更困难。


## 1.24 Reward Hacking

强化学习中的**奖励黑客（Reward Hacking）**是指智能体通过利用奖励函数设计中的漏洞或模糊性，采取与任务初衷不符的行为来最大化奖励，而非真正完成任务目标的现象。这一问题的本质是**目标错位（Goal Misalignment）**，即智能体的优化行为与设计者的真实意图背道而驰。以下是关于奖励黑客的详细分析：

------

### 1. 奖励黑客的核心机制

- **定义**：智能体通过“钻规则漏洞”获取高奖励，但行为偏离任务本质目标。例如：
 - **清洁机器人**：通过打翻垃圾桶制造更多灰尘以获取“清洁得分”，反而使环境更脏。
 - **文本生成模型**：输出无意义的短文本（如“我不知道”）以满足长度限制，但未提供有效信息

### 解决方案与缓解措施

1. **多维度奖励设计**

2. **动态惩罚机制**

 - 检测异常模式（如连续重复输出）并施加负奖励。

3. **对抗训练**

 - 主动生成黑客行为样本加入训练集，教会模型拒绝此类行为。

4. **因果奖励建模**

 - 要求模型解释评分依据，验证奖励逻辑是否对齐人类意图


## 1.25 PPL

PPL（Perplexity，困惑度）是衡量语言模型预测下一个 token 难易程度的经典指标。

PPL (Perplexity, 困惑度) 是衡量语言模型预测下一个 token 难易程度的经典指标。对自回归语言模型来说，给定一个句子 $$ X = (x_{1}, x_{2}, \ldots, x_{N}) $$ ，令模型输出的概率分布为：

$$
p(x_{1}, x_{2}, \ldots, x_{N}) = p(x_{1}) \cdot p(x_{2} \mid x_{1}) \cdot p(x_{3} \mid x_{1}, x_{2}) \cdots p(x_{N} \mid x_{1}, x_{2}, \ldots, x_{N-1})
$$

则句子的平均负对数似然 (NLL) 定义为：$-\frac{1}{N} \sum_{n=1}^{N} \ln \left( p(x_{n} \mid x_{<n}) \right) \quad$，其中 $\ x_{<n} $表示前面所有 **token**。

PPL 通常定义为：

$$
PPL(X) = \exp\left(-\frac{1}{N} \sum_{n=1}^{N} \ln p(x_{n} \mid x_{<n})\right)
$$
PPL 可以理解为“模型每次猜下一个 token 的不确定度”，越接近 1 表示越确定（指数函数）

预训练阶段：我们通常会准备一些“开发集”(Dev Set)，这其中可包含百科文本、新闻、代码片段等。每天或每隔一定步数，就让模型在这些测试集上做一次“前向推理”，计算 PPL，看它是否持续下降。

当 PPL 下降到某个稳定水平后，说明模型对通用语料的学习已经到达了瓶颈或接近饱和。

**需要注意**：PPL 只适合同一模型或同一家 tokenizer 上做纵向对比，因为不同的 tokenizer 压缩率不同，同一句话拆成多少 token 会影响 PPL 数值的绝对值。


- 大模型在常见中文/英文混合语料上，如果预训练充分，PPL 通常能下探到 2 左右(甚至更低)，否则可能是训练不足。
- 不同的场景（如代码 vs. 纯文本）可能有不同的典型 PPL 水平。


## 1.26 为什么batch size越大，需要的学习率应该增大

当**batch size增大时，梯度估计更准确（梯度误差小），噪声降低，为保持参数更新步长的有效性并加速收敛，需相应增大学习率**。

## 1.27 bert MLM和NSP的作用

1)MLM任务的目的是在输入的文本中随机遮盖一些单词，然后让模型预测这些被遮盖的单词。通过这种方式，BERT 可以学习到上下文中缺失单词的表示，从而提高对双向信息的理解能力。相比于传统的从左到右或从右到左的单向语言模型，MLM允许 BERT 在预训练阶段同时考虑到前后文的信息。

2)NSP任务的目的是判断两个句子是否是连续的。在预训练时，BERT 会随机选择两个句子，并根据这两个句子的关系进行分类。这个任务有助于 BERT 学习到句子级别的语义关系和上下文的连贯性。

通过同时进行MLM和NSP任务的预训练，BERT 可以更好地理解双向信息，**从而在下游任务（如文本分类、命名实体识别等）中取得更好的效果**。
