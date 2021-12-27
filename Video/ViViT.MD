# ViViT: A Video Vision Transformer

>
    paper:  https://arxiv.org/abs/2103.15691
    accept: ICCV2021
    author: Google Research
    code(Unoffical):  https://github.com/rishikksh20/ViViT-pytorch
### 一、前言
Google的这篇paper和FBAI的很类似，都是给出几个VideoTransformer的范式，解决怎么把Transformer结构从Image迁移到Video上。相比于TimeSformer，google的这篇paper不单单给出结构范式，同时也给出了迁移imagenet pretrain的实验分析，并给出了怎么训练小数据集上的策略。实验表明，相比于TimeSformer，ViViT更SOTA。(ps:这两篇paper第一版本相差了1个半月的时间，很有可能是同期工作，也有可能Google填了一点FB的坑)

### 二、出发点
基本上使用Transformer作为结构都有一个共性，那就是```self-attention```的特性：
- 长距离依赖性(long-range dependencies)
- 序列中的每个元素都会与整体进行计算

相比于CNN结构，Transformer具备更少的归纳偏置，所以需要更多的数据或者更强的正则来训练。

对于Video数据，要满足时序和空间特征的计算，所以要改进Transformer结构。

### 三、算法设计
为了让模型表现力更强，ViViT讨论了两方面的设计和思考(TimeSformer只考虑了模型结构设计)：
- Embedding video clips
- Transformer Models for Video

先看一下ViT(Image)的公式定义：
输入为$x_{i}\in \mathbb{R}^{h\times w}$, 经过线性映射后，变换为一维tokens，$z_{i} \in \mathbb{R^{d}}$。输入的编码表达如下:
$$
\textup{z} = \left [z_{cls}, \textup{E}x_{1}, \textup{E}x_{2},...,  \textup{E}{x_{N}}  \right] + \textup{p}
$$
这里的$\textup{E}$是2d卷积。如下图最左边结构所示，添加一个可学习的token,$z_{cls}$,用于表达最后分类之前的特征。这里，$\textup{p} \in \mathbb{R}^{N\times d}$表示为位置编码，加上输入token，用来保留位置信息。接下来，tokens经过$L$层的Transformer编码层。每层$l$都包含**Mutil-Headed Self-Attention(MSA)**和**LayerNorm(LN)** 以及一个**MLP**结构，表示如下:
$$
y^{l} = \textup{MSA}(\textup{LN}(\textup{z}^{l})) + \textup{z}^{l}\\
\textup{z}^{l+1}  =  \textup{MLP}(\textup{LN}(\textup{y}^{l})) + y^{l}
$$
MLP由两个线性层以及GELU构成，整个推理过程中，token的维度$d$保持不变。最后，用一个线性层来对encoded的$z_{cls}^{l} \in \mathbb{R}^{d}$完成最终的分类，当然也可以用除了cls-token以外的所有tokens的的全局池化来进行分类。
![ViT](https://ae05.alicdn.com/kf/Hcc300c2f524343d49917925c8913a527c.png)


