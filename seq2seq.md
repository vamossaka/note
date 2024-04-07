# seq2seq
## 编码器-解码器架构
这种架构导致我们需要在encoder和decoder两处进行序列输入。encoder输入被翻译文本，decoder输入真正的翻译出的文本。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_16-27-29.png)
> encoder和decoder间的联系，靠encoder将其中的RNN最后一个时间步的隐状态传给decoder

``` python
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
        # 输出'X'的形状：(num_steps,batch_size,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```
> encoder代码
```py
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
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```
> decoder代码
## 训练过程
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_16-39-24.png)
> 解码器，一个一个读取标准答案的单词作为输入（强制教学）

在上面的循环训练过程中，如图9.7.1所示，特定的序列开始词元\<bos>和原始的输出序列（不包括序列结束词元\<eos>）拼接在一起作为解码器的输入。
除此之外，每次输入还有encoder传来的隐状态和decoder自己学习的隐状态。
## 预测过程
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_16-39-39.png)
> 编码器的工作过程与训练时一样，仅负责提供隐状态，不产生输出结果

与训练类似，序列开始词元\<bos> 在初始时间步被输入到解码器中，但在预测过程中，没有正确译文作为解码器的输入，当前时间步的输入都将来自于前一时间步的预测词元。该预测过程如 图9.7.3所示， 当输出序列的预测遇到序列结束词元\<eos>时，预测就结束了。

## 选择这种架构的原因
虽然都是预测类问题，但不同于之前RNN中的文本续写问题，翻译这种问题需要使用两种不同的vocab。