# RNN
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_11-25-17.png)
> 一种经典的神经网络，带有记忆力
## SimpleRNN
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_11-25-45.png)
> 设计思路：加上一个隐藏层，将隐藏层的数据存入一个memory，在下一次数据输入时，与memory的值一块进行计算（因此输入数据的顺序不同，即使同一个输入数据，输出值也不同）

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_11-20-49.png)
> 神经网络图示（该图中不是三个网络，而是同一个网络被用了三次）

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-11-03.png)
> 也可以是多层网络

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_11-21-44.png)
> RNN的变式
 Elman就是上面介绍过的
 而Jordan的是将output的结果放入memory中

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-18-33.png)
> 双向的RNN，可以考虑正向和反向两侧的数据，考虑的范围更广，得到的信息更多

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-20_22-42-31.png)
> 具体实现

## GRU
> 引入LSTM前先介绍门控单元GRU

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-20_22-41-46.png)
> 具体实现
权衡对当前和过去状态的考虑程度

## LSTM
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_10-06-12.png)
> 三个门的打开与否，都是网络自己学到的

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-41-15.png)
> 数学原理

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-40-45.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-42-27.png)
> LSTM的block可看作复杂化的neuron，一个block对一个变量从原来对应一个参数升级成四个参数

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-46-53.png)
> 左侧的block传递数据给右侧的

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-09_12-47-58.png)
> 多层LSTM

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-20_22-49-46.png)
> 不同于之前的，我们在隐藏层里面又引入了了一个记忆单元C
F决定考虑之前状态的程度，I决定对当前状态考虑的程度
O决定H考虑C的程度 

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-20_22-40-44.png)
> 具体实现

## RNN的应用实例
**我们做一个文本续写的预测问题**

- 在训练时，喂给模型的数据是一个维度为（batch_size，num_steps,vocab_size）的张量，意味着一个batch训练多段句子。
在训练时每走一个时间步，读取对应位置的单词，产生对应位置的单词，走完了一个时间步，即可产生相同时间步的句子。
- 而在给定一段句子，需要模型进行预测时，模型每次仅仅读入一个单词，先进行预热，走完原句后，开始一个一个吐单词。