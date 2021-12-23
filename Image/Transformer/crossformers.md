# 浅谈CrossFormer

> 论文名称: CROSSFORMER: A VERSATILE VISION TRANSFORMER BASED ON CROSS-SCALE ATTENTION
    论文链接: https://arxiv.org/pdf/2108.00154.pdf
    论文代码：https://github.com/cheerss/CrossFormer

# 1. 出发点
Transformers模型在处理视觉任务方面已经取得了很大的进展。然而，现有的vision transformers仍然不具备一种对视觉输入很重要的能力：**在不同尺度的特征之间建立注意力**。
- 每层的输入嵌入都是等比例的，没有跨尺度的特征；
- 一些transformers模型为了减少self-attention的计算量，衰减了key和value的部分特征表达。

# 2. 怎么做
为了解决上面的问题，提出了几个模块。

1. Cross-scale Embedding Layer (CEL)
2. Long Short Distance Attention (LSDA)
3. Dynamic Position Bias (DPB)

这里1和2都是为了弥补了以往架构在建立跨尺度注意力方面的缺陷，3的话和上面的问题无关，是为了使相对位置偏差更加灵活，更好的适合不定尺寸的图像和窗口。这篇文章还挺讲究，不仅提出两个模块来解决跨尺度特征attention，还附送了一个模块来搞一个搞位置编码。

# 3. 模型结构

![模型结构](https://tva1.sinaimg.cn/large/008i3skNgy1gt537h4zcnj31m00hu0x8.jpg)
模型整体的结构图如上所示，与swin-transformers和pvt基本整体结构一致，都是采用了层级的结构，这样的好处是可以迁移到dense任务上去，做检测，分割等。
整体结构由以下组成：
1. Cross-scale embeeding layer (CEL) , 用来做patch embeeding和patch merging(下采样)。
2. CrossFrom block， 看上图(b)，整体看是两个transformer结构的block所组成，其中第一个transformer block采用的是SDA，也就是short distance attention,并且引入了一个DPB模块，第二个transformer block采用的则是LDA，也就是long distance attention，同样也引入了一个DPB模块，两个transformer block串行，组成一个CrossFormer block。
3. Classification Head， 就是常规的分类MLP，没啥可说的。

## 3.1 Cross-scale embeeding layer (CEL)

### Q&A
**Question**:既然是层级结构，那么就一定会有尺度上的下采样，那crossformers是怎么做的呢？
**Answer**: 简单回顾一下pvt和swin的做法
*pvt*: 假设feature map为$B \times H \times W \times C_{1} $, 那么我们就可以做一个stride为2的一个convolution, 变换为$B \times H //2 \times W //2 \times C_{2}$，由于patchsize固定，所以，featuremap下采样，对应的就是token的下采样。
*swin*: swin由于是基于windows做attention，为了达到下采样的效果，选择直接对featuremap上采样，每个4邻域都会分别采样到另一个map里面去，最后则有$B \times H \times W \times C_{1}$变换为$B \times H // 2 \times w //2 \times C_{2}$，也可以看做是stride为2带有空洞的卷积操作。

**Question**: 万变不离其宗，所以为了达到下采样的效果，用*卷积*其实就可以了。那么CrossFormer为了实现下采样是怎么做的呢？
**Answer**: 
![patch embeeding](https://tva1.sinaimg.cn/large/008i3skNgy1gt62f4s2vej30sq0lqju8.jpg)
看上图，很明显，直接用不同卷积核来对输入的图片做卷积，得到卷积后的结果，直接concat一起，作为我们的patch embeeding。想法很简单，实现的话也很朴素，通过不同卷积核的卷积，来获取不同尺度特征的信息，对于变化尺度的物体相对来说是比较友好的，这个可行性其实在很多paper里面都有用到过，比如**Pyramidal Convolution**, 如下图所示。
![尺度卷积](https://tva1.sinaimg.cn/large/008i3skNgy1gt65uye82bj30yq0g076c.jpg)

**ps**: 这里除了patch embeeding，也就是第一个CEL用的是4个卷积核stride为4来做多尺度，其余的CEL也就是patch merge用的都是2个卷积核stride为2来做的多尺度。两个操作基本相同，只看一份代码即可，核心代码如下:
```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        ...

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x
```
代码做了两件事情:

- 初始化几个不同kernel，不同padding，相同stride的conv
- 对输入进行卷积操作后得到的feature，做concat

这样, 以输入为224x224为例, 我们通过patch embeeding, 得到了一个56x56的featuremap，输入到第一个stage，输出继续做一个patchmerging，得到了一个28x28的featuremap，输入到第二个stage，输出继续做一个patchmerging，得到了一个14x14的featuremap, 输入到第三个stage, 输出再次做一个patchmerging，得到一个7x7的featuremap，在输入到最后一个stage，最后的输出做分类即可，基本上都是这么一个套路了，大同小异。那么stage里面是怎么做的，看下一节。


## 3.2 Stage block

对于标准的transformerblock来说，假设输入为$B \times N \times L $, 经过transformer后，我们的输出还是$B \times N \times L$，输入和输出是没有变化的，唯一的尺度变换都在patch embeeding和patch merging。那么我们在改动transformer block的时候，也是要遵守这一原则，对应的，如果想有resolution上的变化，那么就要借助于reshape或者view等操作，好了，不说废话，看这篇文章的crossformer block是怎样的。

### CrossFormer Block
![cfblock](https://tva1.sinaimg.cn/large/008i3skNgy1gt695g6pixj30ga0em752.jpg)

CrossFormer block由两个transformer的block堆叠而成，两个transformer block的self-attention都是基于windows来做的，不同之处在于一个考虑的是局部内的信息，一个则是考虑的是全局的信息。这个思想并没有什么突出的地方，目前来说transformer做局部和全局的串联，已经屡见不鲜。

### Q&A

**Question**: 问题来了，怎样实现呢，既要保证基于windows做self-attention，又想要全局的信息？
**Answer**: 使用一个固定的步长step，比如2或者3，对行和列分别按步长采样，这样可以得到多个全局的信息，同时基于一个$\frac{H}{step} \times \frac{W}{step}$大小的windows。这样最大可能的利用到了featuremap的全局性，同时节省了计算的复杂度，假设输入为$S \times S$，step为$I$，那么windows的窗口大小为$G \times G, G= \frac{S}{I}$，原始的复杂度为$O(S^{4})$, 那么基于窗口的attention的复杂度为$O(G^{4}) = O(G^{2}(\frac{S}{I})^2) == O(G^{2}S^{2}), G<S $。

### CrossFormerblock中的基石: windows self-attention 

- **Short Distance Attention(SDA)**
    ![SDA](https://tva1.sinaimg.cn/large/008i3skNgy1gt6bc96h6ij31iu0duwgl.jpg)
    对于一个$4 \times 4 \times C $的feautremap，如果我们想要实现self-attention, 需要先转换为$(4 \times 4 ) \times C$的向量，那么这里就是所谓的long-range的attention，也就是全局的。但是对于MHA来说，部分head还是更多的focus到short-range，结合swin和twins的结论可以验证，局部attention不仅可以达到很好的效果同时还会节省计算。那么怎么获得局部的attention，很简单，如上图所示，只需要把原始的$4 \times 4 \times C$做reshape操作, 既可以得到$4 \times (2 \times 2) \times C $，那么我们只需要对4个$2 \times 2$做attention即可，最后在reshape回原始形状，代码如下:
    ```python 
    x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
    x = attention(x)
    ....
    ``` 
- **Long Distance Attention(LDA)**
    ![LDA](https://tva1.sinaimg.cn/large/008i3skNgy1gt6bej89jgj31jg0dwwgl.jpg)
    从上面的SDA, 我们得到了局部attention，但是也说了，部分head是局部友好的，也就是说，对于self-attention来说，long-range始终是必不可少的，所以还是需要引入long distance attention。如上图所示，颜色一致的部分表示的是归属于同一个sub-windows的，对于原始的$4 \times 4 \times C$，使用step为2进行采样，得到了4个$2 \times 2 \times C$, 可以抽象成两种计算方法，一种是空洞卷积，一种则是1x1的卷积，stride为step，对于图像来说，相邻的位置，像素所表达的信息接近，所以两种得到的都是全局的一个感受野，所以对应我们的attention，也会得到一个近乎全局的attention，代码如下:
    ```python
    x = x.reshape(B, G, H // G, G, W // G, C).permute(0, 2, 4, 1, 3, 5)
    x = attention(x)
    ...
    ```
直接看这个代码可能不太好理解，我们用```einops```简单改写一下，代码如下:
输入：
```python
x[0,:,:,0]
tensor([[ 1,  2,  3,  4],
        [ 2,  4,  6,  8],
        [ 3,  6,  9, 12],
        [ 4,  8, 12, 16]])
x.shape
torch.Size([1, 4, 4, 1])
```
SDA:
```python
a1 = rearrange(x, ' b (h g1) (w g2) c -> b h w g1 g2 c ', g1=2, g2=2)
a1[0,:,:,:,:,0]
tensor([[[[ 1,  2],
          [ 2,  4]],

         [[ 3,  4],
          [ 6,  8]]],


        [[[ 3,  6],
          [ 4,  8]],

         [[ 9, 12],
          [12, 16]]]])
```
对于SDA的情况，实际上就是循环HW，扣2x2的区域下来，那么因为有行遍历优先，或者列遍历优先，实际上得到的结果是顺序的。
LDA:
```python
a2 = rearrange(x, ' b (g1 h) (g2 w) c -> b h w g1 g2 c ', g1=2, g2=2)
a2[0,:,:,:,:,0]
tensor([[[[ 1,  3],
          [ 3,  9]],

         [[ 2,  4],
          [ 6, 12]]],


        [[[ 2,  6],
          [ 4, 12]],

         [[ 4,  8],
          [ 8, 16]]]])
```
那么对于LDA的情况，我们希望的是外循环是有间隔的，所以把step放到HW的外面，这样循环的时候则是按间隔来进行sample，以达到全局的效果。

### CrossFormerblock中的位置编码: Dynamic Position Bias(DPB)

- **Relative position bias (RPB)**
    随着位置编码技术的不断发展，相对位置编码偏差逐渐的应用到了transformers中，很多的vision transformers均采用RPB来替换原始的APE，好处是可以直接插入到我们的attention中，不需要很繁琐的公式计算，并且可学习性高，鲁棒性强，公式如下:
    $$
        Attention = Softmax(QK^{T}/ \sqrt{d} + B)V
    $$

    **Q&A**
    **Question**:但是这里有个问题，对于$Q,K,V \in \mathbb{R}^{G^2 \times D}$来说，会有一个偏差$B \in R^{G^2 \times G^2}$, $B$所表达的则是matrix上的i和j的相对位置的embeeding，很显然，如果图像的尺寸变化，那么可能会超出B所表达的范围，会导致PE没有作用，那么要怎么改进呢？
    **Answer**:很简单，插值或者切片不就好了，但是切片会导致pe完整性差，损失信息，插值是通过原始的位置信息来模拟出来信息，实际上还是原始的信息，没有信息收益。那本文想到的一个方法就是可以通过学习得到位置信息。
    
- **Dynamic Position Bias (DPB)**
    ![DPB](https://tva1.sinaimg.cn/large/008i3skNgy1gt6y8irypvj30e60hegm8.jpg)
    举个栗子，如果我们的窗口大小为$7 \times 7$, 那么我们希望的相对位置范围为$x \in [-6, 6]$假设我们不考虑截断距离，如果我们的窗口突然放大到了$9 \times 9$，那么我们实际的相对位置所表达的信息只是中间的一部分窗口，失去了对外层数据位置的访问。DPB的思想则是，我们不希望通过用实际的相对位置来做embeeidng，而是希望通过隐空间先对位置偏差进行学习，如上图所示。
    DPB，由3个线性层+LayerNorm+ReLU组成的block堆叠而成，最后接一个输出为1的线性层做bias的表征，输入是$(N, 2)$，由于self-attention是由多个head组成的，所以输出为$(N, 1 \times heads)$，代码如下：
    1. 先得到一个相对位置偏差的矩阵，假设group_size的大小为$7 \times 7$，那么bias的维度为$(169, 2)$
        ```python
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
        # generate mother-set
        position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
        position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2Wh-1
        biases = biases.flatten(1).transpose(0, 1).float()
        self.register_buffer("biases", biases)

        biases:
        tensor([[-6., -6.],
        [-6., -5.],
        [-6., -4.],
        [-6., -3.],
        [-6., -2.],
        [-6., -1.],
        ...
        [ 6.,  4.],
        [ 6.,  5.],
        [ 6.,  6.]])
        ```
    2. 构建索引矩阵, 得到了一个$49 \times 49$的一个索引，从右上角为0开始，向左和向下递增。
        ```python
        coords_h = torch.arange(self.group_size[0])
        coords_w = torch.arange(self.group_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.group_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        relateive_position_index:
        tensor([[ 84,  83,  82,  ...,   2,   1,   0],
        [ 85,  84,  83,  ...,   3,   2,   1],
        [ 86,  85,  84,  ...,   4,   3,   2],
        ...,
        [166, 165, 164,  ...,  84,  83,  82],
        [167, 166, 165,  ...,  85,  84,  83],
        [168, 167, 166,  ...,  86,  85,  84]])
        ```
    3. 初始化DBP模块
        ```python
        pos = DynamicPosBias(64 // 4, 8, residual=False)
        ```
    4. 通过DBP生成bias的embeeding，通过索引矩阵进行取值，最后与attn相加
        ```python
        pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
        # select position bias
        relative_position_bias = pos[self.relative_position_index.view(-1)].view(
            self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        ```
- **Rethinking**: 对于PE来说，目前的形成方法都是通过embeeding来构建bias矩阵，对于VIT来说，直接使用绝对位置的embeeding，通过学习来更新，对于swins来说，直接使用embeeding而不是相对bias的值，相当于$embeeding_{bias} == DBP(bias)$，其实本质上没有太大的差异, 从消融实验结果上来看，DBP和RBP的性能一样。唯一的作用，就是embeeding是后验而不是先验，对于变换的尺寸来说，可能更加友好，只不过这个paper里面没有给出结论，还需要更多的实验来验证。
![DBP&RBP](https://tva1.sinaimg.cn/large/008i3skNgy1gt72rmmv7gj30vu08eabs.jpg)


综上，我们每个stageblock里面，都是由SDA+DBP&LDA+DBP堆叠而成，与swin类似，奇数layer走SDAblock，偶数layer走LDAblock，从结构上来看，先局部attention，再全局attention，有一点点由点到面的既视感。

![模型设计](https://tva1.sinaimg.cn/large/008i3skNgy1gt72xu7pdoj31h00u0n3y.jpg)
与其他的paper大同小异了，设计了4种不同FLOPs的模型，Tiny, Small, Big和Large 用来和其他的模型在同等FLOPs下公平比较。$D$表示的是维度，$H$表示的是attention头的个数，$G$表示的是attention窗口的大小，$I$表示的是滑动窗口的间隔。

# 4. 实验结果

![imagenet](https://tva1.sinaimg.cn/large/008i3skNgy1gt732piy6aj31gf0u0aje.jpg)
CrossFormer都是再224x224的图片大小下进行训练，使用的类似DeiT的训练策略，不过采用了更大的warmup(20个，DeiT是5), 学习率为1e-3, weightdecay为5e-2, 与DeiT不同的是，这里随着模型大小的改变，分别采用了0.1，0.2，0.3，0.5的drop path rate。可以看到，在同等数量级的FLOPs的情况下，CF在imagenet上都取得了SOTA的效果。

![detection&segmentation](https://tva1.sinaimg.cn/large/008i3skNgy1gt737fug10j30u00u47c2.jpg)
可以看到CrossFromer在coco2017上基于RetinaNet架构，也可以达到SOTA的效果，高于Twins模型1.4个ap之多。实例分割则是基于Mask-Rcnn的架构，也是SOTA，超过Swin 1.7个ap。相比而言参数量和FLOPs都更少，性能更好。

![segmentation](https://tva1.sinaimg.cn/large/008i3skNgy1gt74qom762j31k40p2qc6.jpg)
语义分割上，可以看到可以看到最多提升3.3%的MIOU，非常厉害了。

![module](https://tva1.sinaimg.cn/large/008i3skNgy1gt74voek7oj30ow0d0gn3.jpg)
消融实验上，可以看到，当CEL和LSDA一起使用的时候，性能最高。不过这实验也很明显了，CrossFormer参考了PVT和swin的设计思想。使用了LSDA，相比于Swin提升了0.6%个点，设计比swin更加朴实，不错的提升。

# 5. 结论

本文提出了一个新的transformers架构称为CrossFormer。其核心设计包括一个跨尺度嵌入层（CEL）和长短距离注意（LSDA）模块。此外，我们提出了动态位置偏置（DPB），使相对位置偏置适用于任何输入尺寸。实验表明，CrossFormer在几个有代表性的视觉任务上取得了SOTA。特别是，CrossFormer在检测和分割方面有很大的改进，这表明跨尺度嵌入和LSDA对于密集预测的视觉任务特别重要。


> ps: 欢迎大家关注我的知乎：https://www.zhihu.com/people/flyegle


