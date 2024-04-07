# Transformer
## 基本结构
### encoder
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_17-59-00.png)
> 整体流程

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-15_22-35-35.png)
> 一个block的内部结构

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-15_22-31-57.png)
> 结构内部详细过程

residual将原来的feature（向量/对象）与经过self-attention后的feature相加。Layer norm（不同于bn），在单个feature上求mean于deviation。


### decoder
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_18-26-46.png)
> 整体流程
总体上来看，我们给定第一个token作为begin，然后decoder以接龙的形式一个一个生成目标向量

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_18-26-27.png)
> 核心步骤：AT


![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_18-29-18.png)
> AT的Multi self attention机制是接龙现象的原因


![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_18-41-13.png)
> 不使用Multi self attention机制的NAT

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-16_18-54-47.png)
> encoder与decoder产生交互的原因：cross attention

## multi-head attention的过程
![](https://raw.githubusercontent.com/vamossaka/mypic/main/0c016e6fbd3c65402be3536d7eedbbea2c739f1c.png)
```py
#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```
