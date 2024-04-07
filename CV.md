# 计算机视觉入门
## 图像增广
> 翻转，裁剪和改变颜色

图像裁剪的原理：选取原图像的某一块区域，将其中的像素点提取出来，然后根据所需要的图片大小（宽高的像素点个数），进行像素点填充（图片放大便是这个原理）

## 微调
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-25_23-26-31.png)
将已有模型的参数直接迁移
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-25_23-26-18.png)
训练中使用更好的正则化
**正则化：避免过拟合，可以提高模型的泛化能力的方法**
**正则化的方法：**
- **损失函数加入$L_{0}$或$L_{1}$或$L_{2}$正则项**
- **提前停止**

## 锚框
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-26_13-08-33.png)
> 进行目标检测的基本单位，拥有一个编号，蕴含两种信息，是否含有关注的物体，和离真实边缘框的偏移

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-26_13-07-57.png)
> 锚框便是我们要训练的对象，让预测的锚框去不断地逼近真实标注的边缘框

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-26_13-08-17.png)
> 评判框之间相似度的参数

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-26_13-09-01.png)
> 给锚框赋予真实边缘框的方法（参考IoU）

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-26_13-09-16.png)
> 预测时对产生大量的锚框的处理。根据置信度降序，可以去除相似的锚框，仅保留最接近真实边缘框的锚框

## 转置卷积
> 名称来源：卷积的输入$X$和卷积核$K$进行运算可以看作，$X$乘以$K$对应的矩阵$W$得到输出$Y$，**而$Y$与$W^{T}$相乘后得到的结果，便是$Y$和$K$进行转置卷积的输出**

```py
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```
输出为True，可见，超参数一致，转置卷积可以还原卷积的输出结构为输入

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-27_12-51-31.png)
> 转置卷积本质上也是一种卷积

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-27_13-53-51.png)
> 输出尺寸公式

## 空洞卷积
**作用**：在较深的卷积网络中，随着降采样的进行，虽然增加了感受野，但是图片分辨率下降了。空洞卷积的出现使得我们不改变卷积核的参数量，在同样的分辨率下，感受野更大。

我们在空洞卷积里引入了扩张率$r$这一概念，实际的卷积核大小$k_{0}$等于
$$k_{0}=k+(k-1)*(d-1)$$
其中d为填充空洞后，特征间的距离（就是扩张率$r$的定义）
**空洞卷积也有缺点，如果只使用空洞卷积，它的卷积核结构导致会有部分数据不参与计算。**