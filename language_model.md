# 语言模型
## 对文本数据的表示

**首先对文本进行相关定义**
- 一段长文本即一个序列
- 特征和标签都是短文本，即一个序列的子序列

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_17-16-50.png)
特征和标签每次由于都成对存在，所以在每个batch里，我们把他们分别放入X和Y这两个张量。每个里面含有多个特征或标签，个数即batch_size的大小

## 对文本数据的切割
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_17-29-13.png)
> 如果在序列的起始点的偏移量选择不同，即使相同的时间步，最终也会切出完全不同的子序列，供我们选择，这为我们提供了极大的随机性

## 对文本数据的选择
### 随机采样
随机采样的结果如上图所示，相邻的batch里第i个特征不能保证在文本里是连续的

### 连续采样
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-21_17-25-35.png)
> 相邻的batch里第i个特征在文本里是连续的

**注意：在数据选择上，不同的batch在相同的位置i上的子序列，属于同一种切割方式**
