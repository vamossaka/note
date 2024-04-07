# Self Attention
> 针对于输入为一组向量的情况，且向量间互相影响
## 引入
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-34-52.png)
对于一个模型的输入，有时可能不仅是一个向量，而是一个未知长度的向量数组。对应的，输出不再是一个简单scaler或class

往往，一句话，一段语音，一个关系网，都可视为一组向量

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-39-55.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-40-20.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-40-45.png)
> 以上三图展示了需求的不同，输出可以有不同的长度
## 设计原理
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-51-33.png)
对于一组输入，有时不能单独只看一个向量，需要看其他向量来确定对象在当前环境下具体的意义。
我们可以选择用window框住多个向量，但是长度可能会出现太长的情况。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_11-04-08.png)
> 使用self attention后，进入全连接层的输入从原来单个普通的向量变为，联系全局其他的向量后，带有语境意义的单个向量

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_11-10-00.png)
> 对于其他向量对单个向量的影响，我们引入了描述相关性的$\alpha$（attention score）这一概念

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-49-53.png)
> $\alpha$的计算方法

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_10-50-58.png)
> ${\alpha}'$是$\alpha$经过softmax得到的，如图便得到了${a}'$经过self attention后对应的${b}'$

## 计算原理
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_19-55-49.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_19-56-22.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_19-57-20.png)
> 由上面三个图可知，对于某一同类变量的求取，选用矩阵乘法的方法，对所有向量对应的该变量进行统一求取，各自对应不同列

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_14-20-22.png)
> 整体流程，需要学习的参数仅仅在三个矩阵中

### multi-head self attention
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-23-51.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-33-01.png)
在考虑向量之间的相关性时，可能是多种的。我们便把一个q，k，v矩阵拆成多个（图中为两个），每一个代表某一层面的相关性，分别进行O的计算，最后进行合并。
### positional encoding的引入
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-27-46.png)
> 在普通的自监督机制里，所有向量在网络看来都是平等的，但是有时候因为上下文语义的关系，位置信息也是需要考虑的因素，便引入了位置编码

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-22_11-10-19.png)
我们使用了一个位置嵌入矩阵表示n个次元的位置信息，维度与词元的维度相同。每个词元占据一个行，在不同列上有着不同的位置信息公式。这样每个词元都拥有了一个独一无二的绝对位置信息。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-22_11-10-40.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-22_11-10-31.png)
> 由三角函数和热力图可知，在低维（列数较小）时，词元在该列上值的变化幅度更大

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-22_11-24-31.png)
> 这种编码方式，也可以让我们对于某个位置词元，乘以一个投影矩阵，进行指定位置的相对位置转换。进而让模型获取到相对位置信息


## self attention的应用
### 音频
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-39-16.png)
> 处理音频，有时候因为数据量过大（1秒中有100个向量），在进行相关性计算时，只考虑了临近范围

### 影像
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-39-52.png)
> 处理影像，在一个pixel上所有channel的数据看作一个向量（相关网络有GAN）

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_19-37-42.png)
> 与专业影像处理的CNN相比，CNN可以视作self attention的特例，receptive filed便是对其他向量需要考虑的范围划分

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_19-39-58.png)
> 因为CNN可视作self attention的特例，self attention更flexible，需要更多的数据防止Overfitting。因此数据量较小下CNN效果更好

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-49-23.png)
> 与RNN相比，自监督拥有可并行的优越性（因为RNN中后面的向量必须等待前面向量传来的记忆）

### 图
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-08_20-49-56.png)
> 自监督对于图具有天然的适应性，不需计算attention score，根据节点间的关系，自然可以得出Attention Matrix，也引入了GNN
