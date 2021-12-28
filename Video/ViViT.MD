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

#### Overview of Vision Transformers (ViT)
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
<center style="font-size:14px;color:#0C00C0;">图1. 整体结构</center> 

#### Embedding video clips

文中考虑了**两种**简单的方法来构建从视频$\textup{V} \in \mathbb{R}^{T\times H\times W\times C}$到token序列$\widetilde{\textup{z}} \in \mathbb{R}^{n_{t} \times n_{h} \times n_{w} \times d}$的映射。还记得在TimeSformer里的token序列吗，实际上是先把$T$与batchsize进行合并，然后用2d卷积进行```TokenEmbeeding```，然后通过交换$T$维度以及额外的```TimeEmbeeding```来实现空间和时序的信息特征交互。这里稍微不同在于，本文的一种新的采样方法是直接在```Embeeding```阶段就把时序的token引入进来了，看下面提出的两种方法。

![uniform](https://ae04.alicdn.com/kf/He084a10e448a429aab5184a9a854a5b5k.png)
<center style="font-size:14px;color:#0C00C0;">图2. 均匀帧采样</center> 

- **Uniform frame sampling**

    如图2所示，最直接的方法就是均匀采样$n_{t}$帧，每帧独立计算```token-embeeding```,并把这些token直接concat起来，公式可以表达为:
    $$
    \textup{concatenate}(token_{i}), i\in [1, T]
    $$
    ViT的token获取是没有重叠的，所以可以直接想象成，我们先把$T$帧的图像拼接起来成一个大的图像，然后用2d卷积得到token，等价于上述表达。所以可获取token序列为$(n_{t}\cdot n_{h}\cdot n_{w})\times d$。

    ![tubelet](https://ae02.alicdn.com/kf/H2a70e3d8301047e0bab3c02d1b7e99e3Y.png)
    <center style="font-size:14px;color:#0C00C0;">图3. 管道编码</center> 

- **Tubelet embedding**
    图3给出了两一种编码方法，同时获取时序和空间的token，实际可以用3D卷积来实现。对于维度为$t\times h\times w$的tubelet来说，有$n_{t}=\lfloor \frac{T}{t} \rfloor, n_{h}=\lfloor \frac{H}{h} \rfloor, n_{w}=\lfloor \frac{W}{w} \rfloor$。相比于第一种方法需要在encoder阶段融合时序和空间信息，这个方法在生成token阶段就进行了融合，直观上看，没有"割裂"感。

#### Transformer Models for Video
文中给出了三种模型变体的范式，如图1右边所示，下面给出详细介绍
- **Model 1: Spatio-temporal attention**
    这个其实没有更改模型结构，和第一篇讲的TimeSformer的```Joint Space-Time```基本一致，合并token $(b,(t\times h\times w), d)$直接送入ViT，由于```self-attention```性质，从第一层开始，时序和空间信息就进行了交互直到最后的分类。这个方法比较暴力也很简单，但是对于```self-attention```的计算量会从$O(n^{2})$增加到$O((n*t)^2)$，采样帧越多，计算量越大。所以简单的方案不是最优的方案，还需要考虑其他的结构改进。

- **Model 2: Factorised encoder**
    ![model2](https://ae01.alicdn.com/kf/H0189928d58f443d6aacd15ef577fcac8k.png)
    <center style="font-size:14px;color:#0C00C0;">图4. 分离encoder</center> 
    如上图4所示，这个模型结构包含了两个分离的transformer编码结构。
- **Model 3: Factorised self-attention**
    ![model3](https://ae03.alicdn.com/kf/Habaf35645bd14097b28b7002277af7b9c.png)
- **Model 4: Factorised dot-product attention**







