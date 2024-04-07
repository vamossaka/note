# 随记
> 看到啥写啥
## CrossEntropyLoss
> softmax+log+nllloss

将数据先进行softmax，再对数据取对数，最后将每个label对应的每张图片的对数取出来，再取负值，则为最后的loss（reduction=none）或取出batch中的所有值后去平均（reduction=mean）
## 四种优化方法
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-14_15-17-59.png)
> 动量法

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-14_15-30-46.png)
> AdaGrad

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-14_15-28-13.png)
> RMSProp

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-14_15-24-49.png)
> Adam
## sum()的用法
sum有两个作用，求和和降维
如果out是一个向量而不是标量，则应先对其使用sum()函数，达到对函数表达式降维的效果（而不是对结果进行单纯的数值求和），然后进一步求各个自变量的偏导
## python/pytorch语法细碎知识点
- *在形参和赋值时可代表打包成元组和列表，在实参解包（单个元素）。**在形参打包成字典，实参解包（键值对）。zip函数将多个可迭代对象中每个对应位置元素打包成元组，最后形成列表
- for循环使用enumerate（可迭代对象），在每一次返回一个包含序号和元素的元组
- for循环里，用tqdm包装可迭代对象，在每经过一次循环后，进度条增加，并且对可迭代对象本身没有任何影响
- tensor（1）为标量,tensor（ ［1,2,3］）为一维张量，tensor（［［1,2］,［3,4］,［5,6］］）为多维张量
- （a，b，c，d）是维度为4，dim=0对应a，以此类推。维度大小是该维度位置上的值

## 梯度/反向传播
- 当所有自变量的require_grad都为false，因变量的require_grad才为false。

- 叶子节点的要求（三选一）：
它被一些函数显式地初始化，比如x = torch.tensor(1.0)或x = torch.randn(1, 1)。
它是在张量的操作之后创建的，且所有张量都有requires_grad = False。
它是通过对某个张量调用.detach()方法创建的。
- 节点拥有梯度的要求：
为叶子节点
requires_grad = True
- 节点梯度能被计算的要求：
从因变量到自变量，一路上的所有节点都是requires_grad = True
- 全连接层每一层的输入是函数的参数，w和b是函数的变量，输出就是函数的结果。
简单backward的过程为：从最终输出loss出发，每一层根据函数形式和输入的值和自变量当前值(与输出无关)，求得每一层自变量（比如全连接层的w和b，激活函数中的x）的梯度，最后用链式法则相乘求得所需节点的梯度。

## embedding层
> 只有weight，没有bias和activation

embedding层的意义：高维稀疏的one hot vector作为输入，与嵌入矩阵W相乘后降维，映射到低维连续的向量。并且使用特征，可以理解词语的语义，语义之间相近的词在向量空间也更接近。

## 矩阵数学知识
- 矩阵相乘torch.mm的要求是，前面矩阵的列数等于后面的行数。结果矩阵中（m，n）位置的值等于前面矩阵的第m行每一个元素乘以后面矩阵第n列每一个元素后，再相加。
- torch.cat的方法一般作用于一个矩阵列表，指定维度0或1，最终得到的结果矩阵维度数目不变。若为0，不同矩阵在最终结果堆在不同行；若为1，不同矩阵堆在不同列。
并且cat的使用条件是除了指定的dim以外，其余维度上的数值一样。最终张量的维度上的数值：指定的维度数值等于相加张量在该维度上数值之和；未指定维度的维度数值大小不变。（[2,3,4]+[2,3,6]=[2,3,10]）
- torch.transpose和torch.permute都是转置函数
- torch.bmm将一个batch的矩阵相乘，格式：(b,m,n)和(b,n,h)得(b,m,h)

## 预训练模型
在大量数据集下训练出来的模型，供人们在进行下游任务时调用，在其基础上完成微调

## 无监督学习和有监督学习
有监督学习有预先人工标注的数据作为label，而无监督没有（自监督的ground truth要从自己原始数据中提取）
自监督学习主要包含生成式，对比式和对抗式

## 对比学习
> 可以属于自监督和有监督

拉近与正样本的距离，拉远与负样本的距离
