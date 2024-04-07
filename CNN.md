# CNN
> 专业进行影像分类的NN
## 前置：loss的选择
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_14-12-28.png)
对于分类问题，神经网络的输出不同于regression问题里一个值，而是一个向量（one hot vector）。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_14-13-48.png)
输出的结果还经过了softmax的包装，新的向量的每个分量都在0-1之间。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_14-24-15.png)
在最后的loss函数选择上，抛弃了MSE，而选了cross entropy。

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_14-24-58.png)
从图片的结果看出，想从左上走到右下，cross entropy的坡形更理想。
## CNN的设计原理
### 卷积层概念
#### 简化一
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-03-35.png)
> 在一个神经网络里，如果每一个神经元都要处理整张图片，参数量则会巨大，导致Overfitting

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-07-42.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_18-55-53.png)
> 神经元不再观察整个图片，而是寻找图片中的某个特征。因此对单个神经元的作用范围进行重新划分，仅负责一小块区域

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_18-57-12.png)
> 常见设定如图，并且一块receptive field上有多个neutron监控

**padding的意义：随着卷积过程的进行，feature map越来越小，边界像素点会损失。为了保留边缘信息，引入了padding，在每次卷积前先进行一部分填充，这样原来的信息会保留，损失的只是填充信息。**

**padding的副作用：卷积层数过深，可能会有累计效应**

#### 简化二
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-19-12.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_18-59-40.png)
> 对于同一特征，它会出现在一张图片的不同位置，所以在多个位置上安排负责监控的“相同”的神经元。
我们将不同位置上的神经元设置了相同的参数，便达到了目的

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-31-19.png)
> 常见设定如图：联系简化一里的设定，导致了在一块区域上有一组神经元，内部含有的每个神经元参数相同



### 卷积层的实操
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-54-45.png)
> 一个filter即一个矩阵，矩阵里的数字就对应神经元里的参数

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-51-34.png)
> filter在图片的区域与对应的矩阵进行运算，可得出一个值，这些值又可以构成一个矩阵，称为feature map 

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-47-59.png)
> filter的高度就是扫过区域的channel数，经过一个卷积层后得到的channel数就是filter的数量

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_19-49-36.png)




**对filter的定义：针对某一特征而进行识别的神经元**
- **它把一段区域关于某一特征的相关度用数字来表示，该过程可视为对图片的压缩**
- **一种filter会在图片上不断移动，以此来实现对全局的监控，该过程也被称为卷积**

### 池化层
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_20-20-48.png)
> max pooling会取出卷积后的一块区域里面的最大值，二忽略其他值

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_20-21-40.png)
> 与卷积层的搭配

## CNN的整体流程
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_20-22-17.png)

## CNN的公式
**feature map大小**
$N=(W-F+2P)/S+1$   （向下取整）
> $W$为输入大小，$F$为卷积核大小，$P$为padding大小，$S$为步长stride，$N$为最后输出的大小

**与卷积核相关的大小**
- 输入通道个数=每个卷积核的通道个数
- 卷积核个数=输出通道个数（feature map个数）

*注：1\*1的卷积核主要为了改变通道数（一般降维），不改变高宽；池化层不会改变通道数量，但会改变高宽*

**感受野的计算公式**
![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240401192808.jpg)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240401192818.jpg)
> $j$代表特征图中每个特征的间隔距离，随着卷积的继续，呈指数式增长，且仅与步长有关。
而每一层特征的感受野大小等于前一层的感受野大小加上该层卷积核大小减一后乘上前一层特征距离。
$start$代表感受野的几何半径（与感受野大小不同），指的是感受野的半径范围内，有多少该层的特征。


**nn.Conv2d**的参数
in_channel，out_channel确定了卷积核的通道数和卷积核的个数，stride和padding确定高宽


## 神经网络中层与神经元的理解
- 神经网络每一层由一组神经元构成，神经元代表一个值，全连接的神经元在一层里分布为一列，卷积的神经元在一层里分布成3D的形状。
而参数都存在于神经元之间的连接（参数矩阵和滤波器），我们的卷积网络和全连接网络里，linear和con2d的结构都指的是层间参数的结构。

- 卷积层的参数结构与前后两层神经元个数无关（虽然与输入和输出通道数有关），全连接层的参数结构受前后两层神经元个数制约（因此模型对输入有限制性）

- 神经元的感受野：表示经过卷积得到的图片的某个像素点，映射在原图片上的范围。值越大，代表蕴含更全局，语义层次更高的特征。因此可以用来判断每一层的抽象程度。


## 经典CNN模型
### Lenet和Alexnet
两种模型较为简单，均是基本的卷积+池化。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_19-37-24.png)
> Alexnet稍多的一些特性

### VGG
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_19-51-21.png)
> 提出了块的概念，将卷积层包装成块 

### GoogLenet
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_18-36-15.png)
> inception块的使用可以降低网络的参数数（使用1*1卷积核降低了通道数，高宽不变，总参数减少）

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_18-37-13.png)
> 大体结构

### Resnet
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_17-24-44.png)
> 思路来源：不断在原有的层上嵌套，保证网络可以不断逼近最优解


![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_18-57-37.png)
> residual块的细节实现：后面层的输入不再仅仅是前面层的输出，还包含了前面层的输入

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_19-18-30.png)

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_19-19-13.png)
> 整体流程

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_17-24-05.png)
> 在网络深度很深的情况下也有很好的效果

