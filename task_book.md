# 任务攻略
## 总体思路
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_08-23-31.png)
### Model Bias
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_08-31-51.png)
模型的弹性不够大，在模型能够延申的范围内无法找到真正满足最小loss的模型。可以在模型中选择更多的feature或者选用神经网络来增加弹性。
### Optimization
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_08-43-10.png)
优化失败时，是因为选择梯度下降的优化方法，在达到critical point时，无法再次优化模型。
而此时情况可以细分为两种：
- local minima，此时四周都要更高，无路可走
- saddle point（鞍点），仍可逃离
#### 两者区分方法
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_08-48-51.png)
> 对二阶导构成的矩阵进行特征值分析

对模型使用泰勒展开，因为都是critical point，所以一阶导（向量）的项可删去，仅保留二阶导（矩阵）项。
#### saddle point的逃离
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_08-59-17.png)
取特征值为负时对应的特征向量，带入展开式中，即可减小loss
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_09-16-06.png)
> local minima的真相：当参数较多时，在高维的层面上，低维的local minima可能会转化为saddle point，所以真正意义上的local minima非常少

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_09-16-51.png)
> 针对这种情况，我们引入了minimum ratio这一参数，用来表示正特征值在总特征值的比例，比例越高，代表越接近local minima，真正意义上无法优化
### Overfitting
> 在training时loss很小，但是testing时loss大

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_09-29-18.png)
> 更有弹性的model更可能overfitting

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_09-32-00.png)
> 解决方法1：增加数据量/数据处理

![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-05_09-36-36.png)
> 解决方法二：增加模型限制

**对于Model Bias和Overfitting有着模型弹性的trade-off，通过validation set选出最合适的model**

### Mismatch
![](https://raw.githubusercontent.com/vamossaka/mypic/main/Snipaste_2024-03-06_08-52-32.png)
> 训练集和测试集的数据分布不同