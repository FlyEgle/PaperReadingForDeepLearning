# Is Space-Time Attention All You Need for Video Understanding?

```text
paper:  https://arxiv.org/abs/2102.05095
accept: ICML2021
author: Facebook AI
code(offical):  https://github.com/facebookresearch/TimeSformer
```
### 一、前言 
Transformers(VIT)在图像识别领域大展拳脚，超越了很多基于Convolution的方法。视频识别领域的Transformers也开始'猪突猛进'，各种改进和魔改也是层出不穷，本篇博客讲解一下FBAI团队的**TimeSformer**，这也是第一篇使用纯Transformer结构在视频识别上的文章。

### 二、出发点
- **Video vs Image**
    1. Video是具有时序信息的，多个帧来表达行为或者动作，相比于Image直接理解pixel的内容而言，Video需要理解temporal的信息。
- **Transformer vs CNNs**
    1. 相比于Convolution，Transformer没有很强的归纳偏置，可以更好的适合大规模的数据集。
    2. Convolution的kernel被用来设计获取局部特征的，所以不能对超出'感受野'的特征信息进行建模，无法更好的感知全局特征。而Transformer的```self-attention```机制不仅可以获取局部特征同时本身就具备全局特征感知能力。
    3. Transformer具备更快的训练和推理的速度, 可以在与CNNS在相同的计算下构建具有更大学习能力的模型。(这个来自于VIT)
    4. 可以把video视作为来自于各个独立帧的patch集合的序列，所以可以直接适用于VIT结构。
- **Transfomrer自身问题**
    1. ```self-attention```的计算复杂程度跟token的数量直接相关，对于video来说，相比于图像会有更多的token(有N帧), 计算量会更大。


### 三、算法设计
Transformers有这么多的优点，所以既要保留纯粹的Transformer结构，同时要修改```self-attention```使其计算量降低并且可以构建Temporal特征。

#### 构建VideoTransformer
我们先梳理一下Video怎么输入到Transformer中: 对于Video来说，输入为$X \in \mathbb{R}^{H\times W \times 3 \times F}$, 表示为F帧采样的尺寸为$H\times W$的RGB图像。Transformer需要patch构建sequence进行输入，所以有$N = HW/P^2$，这里$P$表示的是patchsize大小，$N$表示的是每帧有多少个patch。展开后，可以表示为向量$X(p,t)\in \mathbb{R}^{3P^2}, p=1,...,N, t=1,...,F$。

对输入做Embeeding处理，$z^{(0)}_{(p,t)} = EX(p, t) + e^{pos}_{(p, t)}$, 这里$E\in \mathbb{R}^{D\times 3P^2}$表示为一个可学习的矩阵，$e^{pos}_{(p,t)} \in \mathbb{R}^D$表示一个可学习空间位置编码。相比于Image的```cls-token```，Video的```cls-token```表示为 $z^{(0)}_{(0,0)}$。

Transformer整体包含L层encoding blocks, 每个block的query,key,value表达如下:
$$
q^{(l,a)}_{(p,t)} = W^{(l, a)}_{Q}LN(z^{(l-1)}_{(p,t)}) \in \mathbb{R}^{D_h}\\
k^{(l,a)}_{(p,t)} = W^{(l, a)}_{K}LN(z^{(l-1)}_{(p,t)}) \in \mathbb{R}^{D_h}\\
v^{(l,a)}_{(p,t)} = W^{(l, a)}_{V}LN(z^{(l-1)}_{(p,t)}) \in \mathbb{R}^{D_h}
$$
这里，$a=1,...,A$表示attention heads数量，$D_{h}$表示的是每个head的维度。

相比于Image的self-attention, Video的self-attention需要计算temporal维度，公式表达为:
$$
a^{(l, a)}_{(p, t)} = Softmax\left({\frac{q^{(l, a)}_{(p,t)}}{\sqrt{D_h}}}^{T}\dot \ \left[k^{(l, a)}_{(0,0)} \left \{k^{(l, a)}_{(p^{'},t^{'})}\right\}_{p^{'}=1,...,N \atop t^{'}=1,...,F} \right] \right)\\
s^{(l, a)}_{(p,t)} = a^{(l,a)}_{(p,t), (0,0)}v^{(l,a)}_{(0,0)} + \sum^{N}_{p^{'}=1}\sum^{F}_{t^{'}=1}a^{(l, a)}_{(p,t),(p^{'},t^{'})}v^{(l,a)}_{(p^{'}, t^{'})}
$$
**Note**: 公式里把```cls-token```单独提出来了，这样方便表达空间和时序维度的attention。

合并每个heads的attention后，进行一个线性投影，送入MLP中，同时进行一个残差连接和Image的Transformer没有区别，公式表达如下：
$$
{z^{'}}^{(l)}_{(p,t)} = W_{O}
\left[                 %左括号
  \begin{array}{ccc}   %该矩阵一共3列，每一列都居中放置
    s^{(l,1)}_{p,t}\\  %第一行元素
    \vdots\\
    s^{(l,A)}_{p,t}\\  %第二行元素
  \end{array}
\right]           + z^{(l-1)}_{(p,t)}\\
z^{l}_{(p,t)} = MLP(LN({{z^{'}}^{(l)}_{(p, t)}})) + {z^{'}}^{(l)}_{(p, t)}
$$

最后就是分类层了，取```cls-token```用于最终的分类。
$$
y = MLP(LN(z^{l}_{(0,0)}))
$$

这样，我们就可以得到一个从输入到输出的VideoTransformer的完整表示。知道了怎么输入输出，接下来讨论怎么改进更好的获取temporal特征信息。

#### Self-Attention范式 

为了解决时序的问题，文中提出了几种构建范式，如下图所示:
![TransformerBlock](https://s3.bmp.ovh/imgs/2021/12/7f15dcb8596d4553.png)
- **SpaceAttention(S)**
    这种就是标准的Transformer结构了，不计算Temporal的信息，只计算空间信息。公式可以表达为:
    $$
    a^{(l, a)}_{(p, t)} = Softmax\left({\frac{q^{(l, a)}_{(p,t)}}{\sqrt{D_h}}}^{T}\dot \ \left[k^{(l, a)}_{(0,0)} \left \{k^{(l, a)}_{(p^{'},t)}\right\}_{p^{'}=1,...,N } \right] \right)
    $$
- **Joint Space-Time Attention(ST)**
    这种就是把temporal和空间的token拉伸在一起，计算量会变得很大($O((n+1)^2)$ -> $O((n*t+1)^2)$)。公式表达为:
    $$
    a^{(l, a)}_{(p, t)} = Softmax\left({\frac{q^{(l, a)}_{(p,t)}}{\sqrt{D_h}}}^{T}\dot \ \left[k^{(l, a)}_{(0,0)} \left \{k^{(l, a)}_{(p^{'},t^{'})}\right\}_{p^{'}=1,...,N \atop t^{'}=1,...,F} \right] \right)\\
    $$
- **Divided Space-Time Attention(T+S)**
    相比于前两种，这个变种的attention计算分成了两步，第一步计算Temporal-self-attention，第二步计算Spatial-self-attention，复杂度则会变为($O((n*t+1)^2)$ -> $O((n+t+2)^2)$),每一次计算都会有```cls-token```参与，所以需要+2。公式表达如下：
    $$
    {a^{(l, a)}_{(p, t)}}_{spatial} = Softmax\left({\frac{q^{(l, a)}_{(p,t)}}{\sqrt{D_h}}}^{T}\dot \ \left[k^{(l, a)}_{(0,0)} \left \{k^{(l, a)}_{(p^{'},t)}\right\}_{p^{'}=1,...,N } \right] \right)\\
    {a^{(l, a)}_{(p, t)}}_{temporal} = Softmax\left({\frac{q^{(l, a)}_{(p,t)}}{\sqrt{D_h}}}^{T}\dot \ \left[k^{(l, a)}_{(0,0)} \left \{k^{(l, a)}_{(p,t^{'})}\right\}_{t^{'}=1,...,F } \right] \right)
    $$
    两步独立计算且意义不同，所以Q,K,V需要来自不同的weights，不能共享权重。简单的定义为:
    $$
    {W^{(l, a)}_{Q^{space}}, W^{(l, a)}_{K^{space}}, W^{(l, a)}_{V^{space}}}\\
    {W^{(l, a)}_{Q^{time}}, W^{(l, a)}_{K^{time}}, W^{(l, a)}_{V^{time}}}
    $$
- **Sparse Local Global Attention (L+G)**
    这个attention文章只做了简单的描述，没有给出相关代码实现，这里参考了[Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf)文章，做一个简单的解释。
    **先引入几个概念和图示**
    **Self-Attention**, 左边是self-attention矩阵，右边是对应的相乘关系，复杂度为$O(n^2)$。
    ![transformer](https://s3.bmp.ovh/imgs/2021/12/20be0e9b9b168962.png)
    **Atrous Self-Attention**，为了减少计算复杂度，引用空洞概念，类似于空洞卷积，只计算与之相关的k个元素计算，这样就会存在距离不满足k的倍数的注意力为0，相当于加了一个k的stride的滑窗，如下图中的白色位置。这样复杂度可以从$O(n^2)$降低到$O(n^2/k)$。
    ![Atrous Self Attention](https://s3.bmp.ovh/imgs/2021/12/843b71d7a2b0316b.png)
    **Local Self-Attention**, 标准self-attention是用来计算```Non-Local```的，那也可以引入局部关联来计算local的，很简单，约束每个元素与自己k个邻域元素有关即可，如下图，复杂度为$O((2k+1)*n)$, 也就是$O(kn)$, 计算复杂度直接从平方降低到了线性，也损失了标准self-attention的长距离相关性。
    ![Local Self Attention](https://s3.bmp.ovh/imgs/2021/12/48556ddf05a8c38a.png)
    **Sparse Self-Attention**, 所以有了OpenAI的Sparse self-attention，直接合并Local和Atrous，除了相对距离不超过k的，相对距离为k的倍数的注意力都为0，这样Attention就有了"局部紧密相关和远程稀疏相关"的特性。
    ![Sparse Self Attention](https://s3.bmp.ovh/imgs/2021/12/6ad5e20ebf801466.png)

    回到本文，local-attention只考虑$F\times H/2\times W/2$的patches，也就是每个patch只关注1/4图像区域近邻的patchs，其他的patchs忽略。global-attention则采用2的stride来在Temporal维度和HW维度上进行patches的滑窗计算。与Sparser self-attention不同点在于，Sparse Local Global Attention先计算local后再进行计算global。

- **Axial Attention(T+W+H)**
![不同attention可视化](https://s3.bmp.ovh/imgs/2021/12/88b0254244902dd0.png)


### 四、算法框架
### 五、实验结果
### 六、结论

