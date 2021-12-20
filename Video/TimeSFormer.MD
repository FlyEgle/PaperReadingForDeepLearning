# Is Space-Time Attention All You Need for Video Understanding?

```text
paper:  https://arxiv.org/abs/2102.05095
accept: ICML2021
author: Facebook AI
code(offical):  https://github.com/facebookresearch/TimeSformer
```
### 前言 
Transformers(VIT)在图像识别领域大展拳脚，超越了很多基于Convolution的方法。视频识别领域Transformers也开始'猪突猛进'，本篇博客讲解一下FBAI团队的**TimeSformer**，这也是第一篇使用纯Transformer结构在视频识别上的文章。

### 出发点
1. 相比于Convolution，Transformer没有很强的归纳偏置，可以更好的适合大规模的数据集。
2. Convolution的kernel是用来设计获取短距离空间信息的，所以不能对超出'感受野'的依赖关系进行建模，无法更好的感知全局特征。而Transformer的```self-attention```机制不仅可以获取局部特征信息同时本身就具备全局感知信息的能力。

### 怎么做
### 算法框架
### 实验结果
### 结论

