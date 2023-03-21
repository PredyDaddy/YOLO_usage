# 1. YOLOV3引入
**图来自于江大白的CSDN博客**
![在这里插入图片描述
sb](https://img-blog.csdnimg.cn/3b8ab6d73f0b4f1fb166bb394b6c1493.png)

# 2. 为什么学习YOLOV3？
1. YOLOV3跟V2/V1相比采用了残差的思想，可以说是基于ResNet的思想上重新设计了backbone(DarkNet53)。
2. V1/V2的backbone采用的是VGG类型的结构, 借着学习V3, 复习ResNet以及为什么DarkNet会有更好的精度和速度
3. 在YOLOV3中加入了FPN层, 这是为什么相比于(V1/V2采用的图像金字塔), V3能够涨点的那么多最重要的原因, 学习图像金字塔的发展到FPN层再手写一个FPN层出来
4. YOLOv3采用了一种基于聚类的正负样本匹配方法，相比于基于IoU的方法，可以更好地适应各种目标尺度和宽高比，并提高模型的精度和稳定性。后续的YOLO也沿用了这种思想，很有学习的必要性。

# 3. Backbone的改进: DarkNet53
**上图**

![在这里插入图片描述](https://img-blog.csdnimg.cn/31e5ca984ebe4994892c2a0f439e5eed.png)



## 3.1 Plain Model对比
**下面这张图是简单模型(Plain Model)的对比, 看一下DarkNet19 vs VGG16, DarkNet53 vs ResNet101**
![在这里插入图片描述](https://img-blog.csdnimg.cn/b71b25afec3b4a7b8a01fdec7b8f56f8.png)

1. DarkNet19相比于VGG16具有更少的参数量(7.29Bn VS 30.94Bn), 更快的GPU推理速度（6.2ms vs 9.4ms）
2. 同样是残差结构的ResNet101和Dark53, 也是同样拥有更少的参数量(18.57Bn vs 19.70Bn)和更高的GPU推理速度  （13.7ms vs 20.0ms）

## 3.2 DarkNet19(YOLOV2) VS VGG16
1. 较少的参数量: DarkNet19是一个FCN(全卷积的神经网络), 而且还使用了1x1的卷积进行通道数的缩减, VGG系列使用的都是3x3的卷积核而且通道数没有缩减。
2.  在网络设计中1x1的卷积核通常是功能性的, 例如减少参数和通道数
3. 较快的速度: 也是因为FC层, VGG16最后三个FC层有太多参数, 同样是在ImageNet上做预训练, DarkNet19直接最后一个1x1 1000通道 softmax解决。
4. 精度的保持
- DarkNet19在后期通道也是增长了，而VGG系列最后一个Block的通道数是没有变化的，这也只是可能的一个因素，这也说明了另一件事VGG系列可能出现了过拟合的现象（个人见解）
- DarkNet19中使用了批量归一化（Batch Normalization）来加速训练和提高模型的准确性，批量归一化在训练过程中可以加速梯度的传播和收敛，同时也提高了模型的泛化能力，这对于推理速度的提高也有积极的影响。




## 3.3 残差网络
1. **重点:为什么当时网络深度较浅？DarkNet19, VGG19?**
- 这个疑惑在何凯明大神的ResNet论文中得到了答案，就是对于Plain模型，网络越深，训练误差和测试误差越大，精度可能相较于浅层网络还会有所下降。
- 因为随着网络的加深，会出现梯度的弥散，很多特征会消失在深层网络，导致结果没办法呈现到最佳的效果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a549dd5054614072a9a131486733e6f5.png)

2. 什么是残差连接？为什么对梯度消失有效？带来了什么效果？
- 原理主要是把卷积层前后的特征进行元素相加求和，如果特征是在这其中的某一个卷积后消失的，那么残差结构便可以找回丢失的特征。
- ![image](https://user-images.githubusercontent.com/109494714/226547410-7b6d6b2c-1c29-47a7-a53d-05a82e7d5af6.png)
- 左图为没有使用残差结构的plain模型，右图是使用了残差结构的残差网络ResNet，可以看到使用了残差结构的模型，随着模型深度的增加，误差随着训练的迭代进行也是降低的。
![image](https://user-images.githubusercontent.com/109494714/226547608-f58533ca-98e3-48d6-a0b2-cbb818226b65.png)

## 3.4 为什么作者不直接使用精度类似的ResNet101而是重新设计了DarkNet59?
1. 首先他们在block中都用conv3x3代替了(VGG16, DarkNet19)的MaxPooling
2. 但是通过对比, ResNet中Block是三层卷积而DarkNet的block是2层卷积, DarkNet53在最后分类前是1024通道而ResNet是2048通道
3. ResNet101中的Stage是3, 4, 23, 3, DarkNet中的Stage是1， 2， 8， 4。可以发现DarkNet的设计更加均衡
4. , 而且最后的通道数其实1024是足够的, FC层之前采用较高的通道数是为了能够提取更加丰富的特征信息, 高通道数的卷积层也有利于模型的泛化能力。 这里也是说明其实1024通道是够的不用上到2048通道
5. 所以可以看出来精度高了1个点但是推理速度快乐1.5倍还多

# 4. 图像金字塔发展史
**YOLOV3很重要的一个改进点就是增加了FPN层进来**
![image](https://user-images.githubusercontent.com/109494714/226585967-6ae80adb-f456-4351-a6c9-a8328444d463.png)   

(a). Featurized image pyramid
- 可以看成不同下采样stride提取特征, 每一次下采样提取特征就会输出一个Feature Map, 通过这个Feaeture Map做一个predict
- 注意对于Feature Map的输出, 并不是直接进行全连接层或卷积层+softmax的操作，因为这样会破坏特征的空间结构信息。如卷积或者空间池化等，来提取目标的位置和特征信息，并生成检测框和类别预测。
- 例如DarkNet53用conv + avgpool + softmax  VGG用maxpool + fc + fc + fc + softmax。

(b). Single Feature Map
- 只有一个特征图的输出, 就像是我们平时用的resnet等等, 最后的predict头其实是有比较高的语义, 对于尺度的变化也是有更好的鲁棒性
- 但是缺点是每层金字塔仍然需要获得最准确的结果, 有研究表明, 图像金字塔每个level都有主要的特点
- 任何BackBone都会产生不同大小的特征图, 由于不同深度导致了较大的语义差距。高分辨率特征图具有低层次的特征，这也损害了其对目标识别的表征能力。对于分类不友好，但是对于定位是友好的
- 对于上面这句话的解释就是, 举例: 在第一个特征图有一个类别占了1/4的区间, 再往上提特征, 不好分类,但是对于定位友好的


(c). Pyramidal feature hierarchy
- 这是SSD的金字塔特征层, 跟(a)有点像但是不同, (a)是对一张图用不同的卷积核操作, 而这个是对一张图不停的下采样
- 这样看起确实是很理想的, 但是SSD放弃了使用已有的特征，而是从网络的高层开始构建金字塔，可以理解为他在最上面那个Feature Map的基础上又新增了几个新层(CONV)，从这个地方开始构建的金字塔
- **跟图c不一样**
- 因此，SSD错过了重用特征层次结构的更高分辨率特征的机会。


(d) Feature Pyramid Network(FPN)
- 他在SSD理想的情况下, 每一个Feature Map上采样成跟上一个Feature Map一样大小的然后相加
- YOLOV3是32倍的下采样
- 上采样的方法之一反卷积: https://blog.csdn.net/bobchen1017/article/details/128607544
- 给ResNet 增加FPN和Focal Loss: https://blog.csdn.net/bobchen1017/article/details/128607709

# 5. YOLOV3的图像金字塔(FPN) 
**看图**

![image](https://user-images.githubusercontent.com/109494714/226595548-2d08ebc6-afaa-44f0-a358-7bfcef248862.png) 
![image](https://user-images.githubusercontent.com/109494714/226595561-c5330601-9273-4968-97d7-d7e38c9518b5.png)

1. 简单点说就是上采样完了之后拼接在一起而不是直接Add在一起，这样拼接出来的Feature Map比较大
2. 拼接之后的Feature Map会比单个Feature Map大。但是由于上采样后的Feature Map分辨率较低，而下采样后的Feature Map分辨率较高，拼接之后可以保留更多高分辨率的信息，从而提高检测精度。
3. 同时，由于拼接之后的Feature Map通道数增加，需要更多的计算和存储资源。因此在设计模型时需要权衡精度和效率之间的关系。

# 6. YOLOV3中的Anchor机制
![image](https://user-images.githubusercontent.com/109494714/226603179-9e18b62a-b3b0-4eb0-9d4b-83d30897273f.png)

![image](https://user-images.githubusercontent.com/109494714/226603206-7cce8856-4260-4306-bcbe-5a2a6042f78f.png)

如果输入的是416×416的3通道图像，YOLOv3会产生3个尺度的特征图，分别为：13×13、26×26、52×52，也对应着Grid Cell个数，即总共产生13×13+26×26+52×52个Grid Cell。对于每个Grid Cell，对应3个Anchor Box，于是，最终产生了(13×13+26×26+52×52)×3=10647个预测框。仍然是每个cell输出三个bounding box
其中不同尺度特征图对应的预测框相对预测的目标大小规模也不一样，具体如下：
- 13×13预测大目标
- 26×26预测中目标
- 52×52预测小目标

yolov3只有这个cell是包含了整个目标才会负责预测和训练

YOLOv3中的每个cell只有当其负责的区域（即anchor box所覆盖的区域）包含了整个目标时，才会负责预测和训练。这是为了避免同一个目标被多个cell预测，导致重复计算和低效。如果一个cell负责的区域中包含了多个目标，那么该cell会选择覆盖面积最大的目标进行预测和训练。



# 7. YOLOV3中每一个cell预测的信息
1. 输出85 （80 classes + 4(坐标) + 1(label)）

![image](https://user-images.githubusercontent.com/109494714/226606395-1ad45e3b-7003-4584-a3c5-c5e6a60ee9cb.png)

# 8. YOLOV3中的坐标表示
1. YOLOv3采用直接预测相对位置的方法预测出bbox中心点相对于Grid Cell左上角的相对坐标。直接预测出(tx,ty,tw,th,t0)，然后通过以下坐标偏移公式计算得到bbox的位置大小和置信度:

![image](https://user-images.githubusercontent.com/109494714/226608067-ec252ff8-98a4-48a4-bbef-f8a5a6332e44.png)

![image](https://user-images.githubusercontent.com/109494714/226608092-4c788794-5c0e-40cc-bd5a-5a9bcb001a55.png)
tx、ty、tw、th就是模型的预测输出。cx和cy表示Grid Cell的坐标；

比如某层的特征图大小是13×13，那么Grid Cell就有13×13个，第0行第1列的Grid Cell的坐标cx就是0，cy就是1。pw和ph表示预测前bbox的size。bx、by、bw和bh就是预测得到的bbox的中心的坐标和宽高。

在训练这几个坐标值的时候采用了平方和损失，因为这种方式的误差可以很快的计算出来。

注：这里confidence = P(Object)×IoU 表示框中含有object的置信度和这个box预测的有多准。也就是说，如果这个框对应的是背景，那么这个值应该是 0，如果这个框对应的是前景，那么这个值应该是与对应前景GT的IoU。

# 8. 正负样本匹配
1. 