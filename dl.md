# ML入门概念
## 机器学习基本三部曲
分为以下三步
- 给出模型
- 定义loss
- 不断优化loss，使其减小
![](https://raw.githubusercontent.com/vamossaka/mypic/main/note1%20pic1.jpg)
### Model定义
> 建立模型，即就是根据问题的表述，写出一个数学公式来求解

![](https://raw.githubusercontent.com/vamossaka/mypic/main/note1%20pic2.jpg)
未知参数有两种，weight和bias
### loss函数


![](https://raw.githubusercontent.com/vamossaka/mypic/main/note1%20pic3.jpg)
e描述预测y与训练集中真实y的差距，定义方式一般有两种，MAE和MSE。

**交叉熵作为loss的两种情况**
- 单分类问题：样本中仅有一个位置是正标签（1），其余位置全为负标签（0），因此是输出结果是一个多项分布，公式为
![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240411093521.jpg)
> 我们仅考虑让输出结果中，对应正标签位置的预测结果不断逼近1即可，其他输出我们不管（因为无意义）

- 多分类问题：样本中不止一个位置是正标签，代表着需要有多个对象等待被分类，所以多分类问题可看作多个二项分布
对于单个对象，公式为
![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240411093533.jpg)
> 这里考虑到每个位置上可能出现1，也可能出现0。1的情况就要该位置的输出逼近1，0的情况就要该位置的输出逼近0，所以式子考虑了两种情况

整体看，为
![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240411093540.jpg)
### optimization过程
> optimization就是通过对模型参数进行不断修正的，使loss的值越来越小

![](https://raw.githubusercontent.com/vamossaka/mypic/main/note1%20pic4.jpg)
gradient（梯度），即为数学中的导数，根据gradient大小，为0时，我们求取loss最小值。
同时观察图片也可发现，local minima（极小值）时，gradient也为0，但不满足loss全局最小（global minima）的要求，所以要注意区分。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/note1%20pic5.jpg)
**$ \eta $是人工定义的参数，名为学习率（属于hyper parameter）。**
我们可以随机选取值作为w和b的初值，在不断迭代后，w* 和 b*即是loss取最小值时对应的两个参数。
#### 关于梯度下降的修正
##### Momentum的引入
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_10-13-39.png)
> 除过gradient以外，我们还引入了momentum，参与到参数向量的移动

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_10-19-23.png)
> momentum带来的好处：走出local minima，翻过小丘

##### 客制化的学习率
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_08-58-30.png)
在errror surface（类似于等高线图，立体地去看）里，高度差较小的方向上，lr应该更大（加快效率），高度差更大的方向上，lr应该更小（不要剧烈波动，以及跨过最小loss）。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_09-08-02.png)
> $\sigma$的一种取法：均方根
梯度大的时候，则步长减小

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_09-10-53.png)
>  $\sigma$的一种取法：RMSProp
相比上一种方法，对于当前梯度和以前梯度有着不同的权重划分，使得步长的时效性更强，更灵活

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-07_09-13-29.png)
> 随着训练的进行，model越来越接近最佳情况，此时也可以不断减小lr，使得步长减小，减小探索范围
warm up：在开始阶段$\sigma$不精准，不具统计意义。所以将lr设小，在四周进行探索
## 神经网络中的全连接层和激活函数
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-15-27.png)
> Sigmoid函数是对Hard Sigmoid这种折线的平滑拟合，相比于一次函数 $ y=b+wx $ 提供了非线性特征

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-16-00.png)
> 对于形状更为复杂的折线，我们也可以用多个Sigmoid函数拟合

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-16-46.png)
> 如果使用多个feature，则Sigmoid函数再次进化

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-29-42.png)
> 展示了全连接层和Sigmoid的搭配使用，以及整个网络的线代表示

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-41-37.png)
> 多层组合便形成了NN，其中我们的Sigmoid或ReLU被称为neuron

[^forward]:前向传播的函数：输入数据经过每一层处理和转换得到预测输出的过程
## pytorch上的实现
### 数据集处理
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-49-41.png)
将数据放入MyDataset中，并且实现了三种方法
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-52-29.png)
Dataloader包装Dataset，创建一个新的对象，指定了数据的batch_size（将数据分为五组），同时调用Dataset中的_getitem_方法时，会返回指定编号的batch数据。

### 神经网络初始化
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_18-49-58.png)
> 两种方式实现我们的网络


网络继承自nn.module，重写了_init_和forward[^forward]两个方法，指定了具体的全连接层和激活函数。
### 训练所需对象准备
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-50-09.png)
> 初始化阶段，创建了所需的对象

### 训练步骤
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-50-22.png)
在训练过程中，以一次**epoch**作为基本单位，经过多次循环。
循环过程分为以下几步

- 优化器清除之前的梯度，避免在反向传播中引起梯度累积
- 指定张量计算需要的硬件设备
- 开始前向传播
- 计算loss
- 反向传播[^反向传播]计算梯度
- 根据梯度更新模型所有参数
[^反向传播]:运用链式法则从输出层到输入层，逐层传播，计算每个中间变量对参数的梯度 

#### batch的选用
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_19-36-47.png)
> epoch的定义：一次epoch即将所有batch都update完一遍
batch的定义：一次神经网络需要的一组数据

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_09-02-38.png)
> 因为GPU可以进行平行运算，在一个batch数据量不是非常大的时候，计算时间没有显著差异

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_09-24-18.png)
> 因为大batch容量大，updates次数少。当一次update时间差不多时，一次epoch的时间更短

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_09-28-24.png)
> 小batch更加noisy，update次数多，更可能逃出local minima之类的情况

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_09-58-43.png)
> 在测试集上，小batch表现也更好。因为更noisy，使得最后的落点在更平坦的地带（陡峭的会很容易跳出），所以当测试集与训练集出现差异时，差值不会太大

#### feature normalization
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-11_20-02-43.png)
> 特征归一化的原因：因为feature数据分布特征不一致，导致error surface图像有一个方向平坦，一个方向崎岖

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-11_20-08-25.png)
> 所以引入了特征归一化，使得数据接近

#### batch normalization
对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布，强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-11_21-21-45.png)
如果都通过BN，那么不就跟把非线性函数替换成线性函数效果相同了？这意味着什么？我们知道，如果是多层的线性函数变换其实这个深层是没有意义的，因为多层线性网络跟一层线性网络是等价的。这意味着网络的表达能力下降了，这也意味着深度的意义就没有了。所以BN为了保证非线性的获得，对变换后的满足均值为0方差为1的x又进行了scale加上shift操作(y=scale*x+shift)，这两个参数都是通过训练学习到的。

核心思想应该是想找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头使得网络收敛速度太慢

![](https://raw.githubusercontent.com/vamossaka/mypic/main/微信图片_20240311213016.jpg)
> 具体流程

### 验证步骤
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-04_20-36-59.png)
> 代码错误，最后一行不应缩进

将model状态调整到eval，同时要求不计算梯度。
*注意*：此处 **len(x)** 大小为一个batch内的数据量，**len(dv_set.dataset)** 大小为数据集中总数据量。
### 测试步骤
在这一步，计算每个batch的loss，将其放入结果数组中。
