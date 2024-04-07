# GAN
> 是一种无监督学习
## Generator产生的缘由
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_14-59-23.png)
> 特定的输入加上一个简单的数据分布，经过生成器后变为复杂的数据分布

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_15-04-47.png)
> 为什么输出是一个distribution？
当需要神经网络根据输入产生具有创造性的输出（即相同输入，但不同输出）

## generator和discriminator
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_15-14-53.png)
> GAN的核心思想是创造两个模块，generator和discriminator，两者虽具有对抗性质，不过在学习过程中，都不断进化。但人们的训练终极理想，是让generator胜出。
不同于其他模型思想，GAN要求不断迭代训练轮次

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_15-13-45.png)
学习阶段分为两段：
- learning D：固定G，通过G产生虚假的对象，在选取一些真实的对象，放入D中，让D学习鉴别两者，更新D
- learning G：固定D，将输入放入G后，得到输出，再将输出放入D中，得到D的输出scalar（对象得分，越高代表越真实），再根据scalar的值返回来优化G

**两种模型的职责不同**
- generator负责生产“假货”
- discriminator负责鉴别“假货”

**两段训练中，训练目标不同：**
- learning D是为了更好地鉴别G的输出，有与正确答案相比较的过程
- learning G只是为了骗过D，没有与正确答案相比较的过程

## 确定损失的方法
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_23-39-41.png)
> 不同于普通计算loss的方法，要计算/量化数据分布间的差异很困难

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_23-42-56.png)
> 我们选择从数据分布中采样来代替distribution整体

## 训练的目标
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_09-03-00.png)
$V(G,D)$代表着discriminator鉴别两种类的能力，值越大鉴别能力越强
> 取max值代表着当前discriminator训练后得到的分辨能力，求最大值即求JS散度

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-18_23-57-13.png)
> 而我们训练generator的目标也就是：建立一个让当前discriminator分辨能力最弱的generator

## 训练的问题
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_09-05-57.png)
> 对于discriminator的训练，如果使用binary classifier，由于二元数据分布很难overlap，导致js散度一直等于$log2$，看不出来训练的进步

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_09-08-49.png)
> WGAN可以用距离来量化数据分布之间的区别，清晰地展示训练的进步

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_09-15-45.png)
> 对D函数的限定：在真实数据分布上值大，假的数据分布值小；并且函数要够“丝滑”，防止出现无限大的情况，训练无法收敛

## 评价训练后的generator产生图片的指标
首先我们提出了quality这一指标
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_12-58-57.png)
> Quality：我们另外创建一个分类器，将产生的图片放入一个图片分类器中，得分越集中，图片质量越高

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_13-00-22.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_13-02-47.png)
以上两种情况代表着，在训练过程中，generator仅仅为了躲避discriminator的勘测，有可能做出两个的“小伎俩”。
- 产生的图片围绕着某个数据点分布，图片极其相似（mode collapse）
- 虽然分布范围变大，但仍然走不出一个圈子，导致图片特征相近（mode dropping）

**因此我们又对图片提出了diversity的要求**
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_13-03-40.png)
> Diversity：将一堆图片放入分类器中，将最后的得分取评价，最终得分分布越平均，代表多样性越好

## conditional GAN
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_15-22-27.png)
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-19_15-24-31.png)
> 在生成图片时，我们也可以加上一些指令，可以是一段话，也可以是一张图片

discriminator进行勘测时，也应对它输入给generator的指令，以此来保证generator的输出没有仅仅为了骗过discriminator，而输出一些实际不相干的图片。
同时，我们也可以对generator进行监督学习，确保输出不走偏。