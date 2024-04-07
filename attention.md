# 注意力机制
## 注意力机制产生的原因
自主性的与非自主性的注意力提示解释了人类的注意力的方式，下面来看看如何通过这两种注意力提示，用神经网络来设计注意力机制的框架，

首先，考虑一个相对简单的状况，即只使用非自主性提示。要想将选择偏向于感官输入，则可以简单地使用参数化的全连接层，甚至是非参数化的最大汇聚层或平均汇聚层。

因此，“是否包含自主性提示”将注意力机制与全连接层或汇聚层区别开来。在注意力机制的背景下，自主性提示被称为查询（query）。给定任何查询，注意力机制通过注意力汇聚（attention pooling） 将选择引导至感官输入（sensory inputs，例如中间特征表示）。在注意力机制中，这些感官输入被称为值（value）。更通俗的解释，每个值都与一个键（key）配对，这可以想象为感官输入的非自主提示。如图10.1.3所示，可以通过设计注意力汇聚的方式，便于给定的查询（自主性提示）与键（非自主性提示）进行匹配，这将引导得出最匹配的值（感官输入）。

**非自主性和自主性提示有选择性地引导注意力。前者基于突出性，后者则依赖于意识。**
**因此注意力机制的训练意义是，学习出自主性指示和非自主性指示如何结合，最终做出感官输入的选择。**
## 相关数学表达
$q$代表查询，$k$和$v$一个键值对，$a(q,k)$是一个注意力评分函数，代表两者的相似度，$\alpha(q,k)$代表注意力评分函数经过一个$softmax$后得到的注意力权重。
## 注意力评分函数
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_22-28-33.png)
> 图片中的key改成query

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_22-28-48.png)

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_21-25-36.png)
**最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。**

## 带有注意力机制的seq2seq模型
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-22_09-22-09.png)
初始化解码器的状态，需要下面的输入：

编码器在所有时间步的最终层输出,将作为注意力的键和值；

编码器在最后一个时间步的全层隐状态，将作为初始化解码器每一层的隐状态；

编码器有效长度（排除在注意力池中填充词元）。

```py
def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,       enc_valid_lens]
```
> 在上面的代码里，在每一个时间步里，我们将每一时间步的解码器的最后一层隐状态放入注意力层里作为query，此外输入也有编码器传来的输出，同时作为注意力层的key和value。最终注意力层输出的结果与解码器的输入cat后再放入RNN里