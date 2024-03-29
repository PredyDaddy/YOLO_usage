# 1. 图像处理的四大任务
- 分类: 对图像进行分类，知道他属于哪一类。用神经网络对图像进行特征提取，平时做训练只要把同类型的图像放在同一个文件夹下就算标记了
- 定位: 定位出目标在图像对应的位置 ex: (x, y) 中心点  + 宽高
- 检测: 分类 + 定位:  
  - 第一步：使用卷积神经网络进行图像特征的提取
  - 第二部：对卷积神经网络的特征，通过标签来约束进行定位(GT)和分类识别(label)的训练
# 2. 目标检测的难点: 
- 位置任意性（Arbitrary object location）：目标可以出现在图像的任何位置，不同目标之间的位置也可能相差很大，因此需要算法具有在不同位置对目标进行检测和识别的能力。

- 大小多样性（Scale variation）：目标的大小可能因为距离、角度、摄像机参数、目标运动等因素而变化，因此需要算法能够识别不同大小的目标。

- 形态的差异性（Shape variation）：即使同一类目标在不同场景下也可能出现形态上的差异，因此需要算法具有识别目标不同形态的能力，以确保能够准确地检测出目标。

# 3. Two-Stage Method: RCNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/fef58ff1e26c48b897e63b8a385aa287.png)
**搜索算法 + SVM 分类 +坐标回归**
1. 搜索性算法: 
- 选择性搜索（Selective Search）是一种用于区域提取的算法，常用于RCNN（Region-based Convolutional Neural Networks）中的目标检测任务。选择性搜索算法通过分析图像中相邻像素之间的颜色、纹理、大小和形状等特征，将相似的像素分为一组，然后将这些组合成不同大小和形状的区域，作为可能包含目标的候选区域。

- 具体来说，选择性搜索算法首先将图像分成多个小块，然后通过合并相邻的小块来生成不同大小的候选区域。合并的过程基于一种称为相似性度量（similarity measure）的函数，它根据像素之间的颜色、纹理、大小和形状等特征计算相邻小块之间的相似度，相似度高的小块被合并成一个更大的区域，相似度低的小块被保留下来。这个过程不断迭代，直到生成足够数量的候选区域。

- 选择性搜索算法的优点是能够有效地减少目标检测任务中的计算量，避免了对整个图像进行分类的复杂度。在RCNN中，选择性搜索算法可以生成多个候选区域，并将它们输入到CNN中进行特征提取和分类，从而实现目标检测任务。
- 搜索算法本身并没有一个专门的网络，因为它们通常是一种算法或方法，用于解决优化或搜索问题。搜索算法的具体实现通常依赖于编程语言和应用环境。


1. 坐标回归和SVM
- 搜索算法后RCNN使用两个独立的全连接神经网络，一个用于坐标回归，另一个用于支持向量机（SVM）分类。坐标回归网络用于精确定位每个候选区域中的目标边界框的位置和大小，而SVM分类网络用于将每个候选区域分配给不同的目标类别。最终，对于每个候选区域，选择具有最高得分的目标边界框和目标类别。
- 需要注意的是，由于RCNN中的SVM分类器是针对每个类别独立训练的，因此对于每个类别，需要单独运行一次候选区域提取、调整大小、特征提取、坐标回归和SVM分类的过程。这使得RCNN在训练和推理过程中非常耗时，并且无法直接处理多个目标实例或不同大小的目标实例。
- 其次就是R-CNN涉及使用全连接层，因此要求输入尺寸固定，这也造成了精度的降低；最后候选区域需要缓存，占用空间比较大。


# 4. Two-Stage Method: SPPNet
![在这里插入图片描述](https://img-blog.csdnimg.cn/3dda845a327a47398bb9dcf8cd7c579b.png)

- SPPNet（Spatial Pyramid Pooling Network）是一种空间金字塔池化（Spatial Pyramid Pooling）技术与卷积神经网络（CNN）相结合的方法，用于解决输入图像大小不一致的问题。

- 与传统的卷积神经网络不同，SPPNet在最后一层卷积层之后，将整个特征图分割为不同大小的子区域，并对每个子区域进行池化操作。这种空间金字塔池化技术可以使得网络对输入图像的大小和形状不敏感，而且不需要对不同大小的输入图像进行重复训练。具体来说，SPPNet在网络最后一层卷积特征图上执行以下步骤：

  - 将特征图划分为不同大小的子区域。

  - 对于每个子区域，进行最大池化操作，得到固定长度的特征向量。

- 将所有子区域的特征向量按照预定义的顺序拼接在一起，形成最终的特征表示。

- 由于SPPNet中的空间金字塔池化技术可以处理任意大小的输入图像，并且只需要执行一次，因此在训练和推理过程中具有很高的效率。与传统的搜索算法不同，SPPNet中的空间金字塔池化操作是固定的，不需要额外的计算，因此不会增加计算复杂度。

# 5. 两阶段目标检测方法——Fast RCNN
1. https://blog.csdn.net/qq_47233366/article/details/125579620?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167824976216800188526281%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167824976216800188526281&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125579620-null-null.142^v73^insert_down1,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=faster%20r-cnn&spm=1018.2226.3001.4187

# 6. 初探单阶段检测器: YOLOV1思想与流程

## 6.1 YOLO思想
1. 与之前的二阶段检测器相比，YOLO会把整张图作为网络的输入，不像二阶段一样，没有选择性搜索的算法，直接用神经网络对整张图进行特征提取
2. 简单说三步走
- Reize Image Size 
  - 把全部图像处理成448 x 448(YOLOV1要求的输入)
  - 把数据归一化到0-1的浮点数
- 神经网络提取特征
- NMS (Non-max-Supression): 干掉多余的方框

## 6.2  YOLOV1的流程
![在这里插入图片描述](https://img-blog.csdnimg.cn/d2c5b5b3c4e54535a4deca6eb60722db.png)

1. 将输入图像分成一个固定大小的网格，每个网格被称为一个Grid Cell，这个网格的大小是通过将原始图像大小缩放为448x448像素后得到的。整张图像共分成SxS个网格，每个网格的大小是固定的，一般为7x7或者14x14。
2. 对于每个目标，根据标注信息计算出目标中心点相对于输入图像左上角点的偏移量 $\delta_x$ 和 $\delta_y$，然后将偏移量 $\delta_x$ 和 $\delta_y$ 转换为相对于所在网格左上角点的偏移量，用于确定包含该目标的网格以及预测目标位置的具体坐标。如果一个目标横跨了多个网格，则选择包含该目标中心点的网格作为负责预测该目标的网格。

3. 对于每个网格，预测该网格内是否存在目标以及目标的类别信息和位置信息（即边界框的坐标（x、y、w、h）和宽高），同时计算目标置信度（confidence），即该目标存在的概率。


## 6.3 如果这个时候有一个目标，他横跨了好几个cell，那我到底用哪一个cell去预测类别和定位信息呢？
1. 如果一个目标的边界框横跨了多个网格，那么YOLOv1算法会选择包含该目标中心点的网格作为负责预测该目标的网格，并将该目标的位置和类别信息预测为该网格的输出。算法会选择距离目标中心点最近的那个网格作为包含该目标的网格，并将该网格的输出作为该目标的预测结果。
2. 在标准的YOLOv1中，一个目标横跨了三个 Grid Cell，但只有包含该目标中心点的 Grid Cell 负责预测该目标的类别和位置信息，并且只有这个 Grid Cell 被用于训练目标检测模型的分类和定位损失函数。其他两个 Grid Cell 则不参与训练，因为它们不包含目标中心点。


## 6.4 如果一个目标横跨了三个cell，可以按照以下步骤确定包含该目标中心点的网格：
1. 计算目标中心点相对于输入图像左上角点的偏移量 $\delta_x$ 和 $\delta_y$。
- $$\delta_x = \frac{x - x_{top}}{w_{img}}$$

- $$\delta_y = \frac{y - y_{left}}{h_{img}}$$
- 其中，$(x,y)$ 是目标边界框的中心点坐标，$x_{top}$ 和 $y_{left}$ 分别是输入图像左上角点的横纵坐标，$w_{img}$ 和 $h_{img}$ 分别是输入图像的宽度和高度。计算得到的 $\delta_x$ 和 $\delta_y$ 均在 $[0,1]$ 之间，表示目标中心点相对于输入图像的位置。
2. 将偏移量 $\delta_x$ 和 $\delta_y$ 转换为相对于 cell 的左上角点的偏移量，即：
$\delta_{x,cell} = \delta_x - i$，$\delta_{y,cell} = \delta_y - j$ 其中，$i$ 和 $j$ 分别表示包含目标中心点的第一个 cell 的行和列。
1. 如果偏移量 $\delta_{x,cell}$ 和 $\delta_{y,cell}$ 的值都在 $[0,1]$ 之间，则该 cell 包含该目标中心点。
2. 如果偏移量 $\delta_{x,cell}$ 和 $\delta_{y,cell}$ 的值不在 $[0,1]$ 之间，那么就需要选择最靠近目标中心点的 cell 作为负责预测该目标的网格。
3. 需要注意的是，如果一个目标横跨了多个 cell，那么可能会出现多个 cell 都认为包含该目标的情况。在这种情况下，可以选择距离目标中心点最近的那个 cell 作为包含该目标的 cell，并将该 cell 的输出作为该目标的预测结果。

## 6.5 从数据量来看训练的影响: 
在使用YOLOv1进行目标检测时，数据的多样性和数量对训练效果有很大的影响。如果训练集中包含的目标类别比较单一或者样本数量较少，那么可能会导致某些网格无法学习到对应的目标类别信息，从而影响目标检测的准确率。因此，建议在构建训练集时，要尽可能包含更多的目标类别和更多的样本，以便让每个网格都能够被某些类别定位训练到，从而提高目标检测的准确性。

## 6.6. YOLOV1网络结构图
![在这里插入图片描述](https://img-blog.csdnimg.cn/606484b2bebd47758e75765095347e1b.png)
1. 需要注意的是，YOLOv1算法采用全卷积神经网络进行训练，可以实现端到端的目标检测，同时通过特殊的损失函数（YOLO Loss）可以实现对目标位置和类别的同时预测和优化。
2.  虽然 YOLOv1 在其网络的最后添加了两个全连接层（FC），但它仍然可以被认为是一个全卷积网络。这是因为 YOLOv1 的前面部分主要由卷积层和池化层组成，没有使用任何全连接层。

3.  全卷积网络的定义是指整个网络都由卷积层和池化层组成，没有使用全连接层。在这种情况下，整个网络的输出都是一个特征图，可以输入到下游任务中，如语义分割、目标检测等。虽然 YOLOv1 在最后添加了两个全连接层，但这并没有影响其前面的卷积层和池化层的性质，所以仍然可以被视为一个全卷积网络。同时，YOLOv1 最后的全连接层被用来将特征图映射到预测空间，这也是目标检测中常见的操作。
4.  在目标检测领域中，端到端的目标检测意味着将整个目标检测系统作为一个完整的单元进行优化和训练。这个系统包括输入图像、特征提取、目标检测和输出预测结果，其中所有步骤都通过一个统一的模型端到端进行处理。这与传统的目标检测方法不同，传统方法将目标检测任务分为多个子任务，并使用不同的模型或算法对每个子任务进行处理，最终将结果合并。

## 6.7 输出分析
1. 从上面最后的网络图我们知道了最后是输出7x7x30
2. 7x7 之前解释过了，被分割的Gird Cell,训练在标注的时候中心点在自己区域的object
3. 30是每个格子需要输出 2 个边界框（bounding box），因此每个边界框需要输出 5 个值：中心坐标 (x, y)、边界框的宽度 w、边界框的高度 h、以及置信度分数（confidence score）。每个格子还需要输出 20 个类别的置信度分数（confidence score），表示该格子中的目标属于每个类别的概率。
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/6d7e8a56271949ce878420762bb056cd.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/f0a996384f3a41638a1b9a435311452c.png)

## 6.8 损失函数解析
1. 从结构可以知道前面20个是类别，后面10个是定位 + 置信度。不同的东西有不同的损失函数

# 7. YOLO V2 
![在这里插入图片描述](https://img-blog.csdnimg.cn/1fbe570d1e1047e4b2b0f85380d04164.png)

- 之前写YOLOV1比较详细，是因为需要深刻理解目标检测的很多知识点
- 同时也是为了写YOLOV2 
## 7.1 改进一: BN层的增加
1. 下面的结构图中的卷积层不只是卷积层，还有BN层和激活函数层(Convolution + Batch Normalization + LeakyReLU Activation)
- 卷积操作：对输入数据进行卷积运算，提取特征信息。

- 批量归一化操作：对卷积层的输出进行标准化，使其均值为0，方差为1，从而加速网络收敛、降低过拟合的风险，同时也可以提高模型的泛化能力。

- LeakyReLU 激活函数操作：对 BN 层的输出进行激活函数变换，引入非线性映射，从而增加模型的非线性表达能力。

## 7.2 BN 批量归一化数学推到
1. 批量归一化（Batch Normalization，BN）是一种深度神经网络中常用的技术，用于加速神经网络的训练、提高模型的精度以及降低过拟合的风险。BN 的基本思想是对每个批次中的数据进行标准化处理，从而使得输入数据的均值和方差保持稳定，降低了数据分布的变化性，提高了神经网络的训练效率。
2. 具体来说，BN 对每个特征维度上的数据进行标准化处理，使得数据在该维度上的均值为 0，方差为 1。标准化的公式如下：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
3. 其中，$x$ 表示原始输入数据，$\mu$ 表示该批次数据在该维度上的均值，$\sigma^2$ 表示方差，$\epsilon$ 是一个小常数，用于避免方差为 0 的情况。对标准化后的数据进行缩放和平移操作，得到最终的输出数据：$y = \gamma \hat{x} + \beta$
4. 其中，$\gamma$ 和 $\beta$ 是可学习的缩放因子和平移因子，用于调整标准化后的数据的均值和方差。通过对输入数据进行标准化处理，BN 可以使得神经网络中每一层的输入分布更加稳定，减少了数据分布的变化性，从而提高了神经网络的训练效率。因为数据在归一化了之后还是要学习的，所以增加了缩放因子和平移因子，不能让他一层不变
5. 除了加速神经网络的训练外，BN 还可以有效地缓解梯度消失和梯度爆炸的问题，从而使得神经网络的深度可以更加深入，提高模型的精度和泛化能力。此外，BN 还可以作为一种正则化方法，降低模型的过拟合风险，从而提高模型的泛化能力。

## 7.3 图像中的归一化
**输入图像尺寸: [4, 3, 240, 240], 4张240x240x3**
1. BN: 四张图片的通道归一化，例如第一次把四张图片的R通道都拿出来，做归一化。在这里做3次，因为输入是3通道的。
2. LN(Layer Norm): 层归一化是一个图片的全部通道拿去归一化，也就是跟其他图片是没有交集的。这里做4次，因为有4张图片 
3. Instance Norm: 每一个图片的每一个channel单独去做归一化，所以这里是4x3=12次
4. Group Norm: 如果输入是4通道，我们可以每2通道做一次归一化

## 7.4 BN为什么有优势
1. softmax函数两头的斜率(梯度)都快接近于0了，我们通过BN归一化把数据集中在中间，让他们又更多的变化，也不会梯度消失
2. 加速训练过程：当网络的输入分布发生变化时，可能导致神经网络训练缓慢或出现不稳定的情况。而 BN 技术可以通过标准化每层的输入数据，使得输入数据的分布更加稳定，从而使得网络的训练速度加快。此外，由于 BN 可以缓解梯度消失和梯度爆炸问题，使得神经网络的训练收敛速度更快，同时减少了训练过程中的震荡，从而提高了训练的稳定性和精度。
3. 提高精度，也使得神经网络能够更深，更宽的去训练。
4. **YOLO v2通过使用BN层使得mAP提高了2%。**


## 7.5 改进二: 预训练的 Darknet-19
1. 在 YOLOv2 中，预训练的 Darknet-19 模型是在 ImageNet 数据集上训练的，而且是在 224x224 的图像上进行训练的。但是，这个预训练模型不是用来进行分类任务的，而是用来初始化 YOLOv2 模型的参数。

2. 在训练 YOLOv2 模型时，使用了 448x448 的高分辨率样本进行训练，并且在训练前，将预训练模型的参数作为 YOLOv2 的初始化参数。在训练过程中，模型的输入是 448x448 的图像，不需要对分类模型进行微调，因为分类模型已经在 ImageNet 数据集上进行过训练，学习到了很多视觉特征，可以直接用来进行目标检测任务。

3. 使用高分辨率的样本进行训练，可以提高检测器的精度和鲁棒性。同时，在使用高分辨率样本时，为了缓解模型过拟合的问题，采用了数据增强的方式，包括随机裁剪、随机缩放、颜色抖动等。

4. 需要注意的是，使用高分辨率的样本进行训练会增加模型的计算量和内存消耗，导致检测速度变慢。因此，在实际应用中需要根据具体的场景和要求进行权衡，选择适当的输入分辨率和模型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/927406fd851f487c9502c8ed1e8937ea.png)


## 7.6 改进三: Convolution with anchor boxes
1. 在 YOLOv2 中，采用了 Anchor Box 来处理目标的尺度和纵横比变化。与 YOLOv1 不同的是，YOLOv2 采用了 5 个预定义的 Anchor Box，分别对应不同的尺度和纵横比。对于每个 Grid Cell，YOLOv2 都会输出 5 个预测值，分别对应 5 个 Anchor Box 的置信度和偏移量。

2. Anchor Box 的使用可以使得 YOLOv2 更好地适应目标检测任务中的不同尺度和纵横比的物体，从而提高模型的精度和鲁棒性。而且，通过预测偏移量而不是直接预测坐标值，可以使得模型更容易学习到目标检测任务中的特征。

3. 另外，为了使得 YOLOv2 更好地适应目标检测任务，作者在网络结构上也进行了一些改进。具体来说，他们去掉了全连接层，使用 Anchor Boxes 来预测 Bounding Boxes，同时去掉了一个 Pooling 层，这样卷积层的输出就能够具有更高的分辨率，从而提高模型的精度。

4. 使用 Anchor Box 可以帮助模型更好地适应目标检测任务中的不同尺度和纵横比的物体，提高模型的精度和鲁棒性。

## 7.7 anchor box 和 bounding box 
Anchor Box 是在训练数据集上进行聚类生成的，而 Bounding Box 是在目标检测模型中预测得到的。两者的作用和生成方式都不同。

注意, 这里的bouding box不是

## 7.8 改进三: 预测偏移量(Direct location prediction)
1. 在 YOLOv1 中，模型预测的目标框中心点总是在相应的 Grid Cell 中心，因此不能很好地适应物体在 Grid Cell 内部偏移的情况，同时也不能很好地适应不同尺度的物体。

2. 而在 YOLOv2 中，模型引入了 Anchor Box 和预测偏移量的概念，使得模型可以预测物体的中心点不一定在 Grid Cell 中心，而是在 Grid Cell 内的任意位置，从而更好地适应物体在 Grid Cell 内部的偏移情况。同时，通过预测边界框的偏移量和大小，模型可以更加精确地预测物体的边界框位置和大小，使得检测结果更加准确和贴合实际。
3. 下图中的tx, ty, tw, th, t0是预测出来的东西 pw，ph是最后选择的那个bounding box的东西。![在这里插入图片描述](https://img-blog.csdnimg.cn/bea7082af3c44ba3ad8e60648bcf3fc9.png)
## 7.9 Fine-Grained Features（细粒度特征）
![在这里插入图片描述](https://img-blog.csdnimg.cn/52b66a7e75ec4d05a7b0e42183afaafd.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0121c2935457482ba6ff4cfaf6c2f381.png)

## 7.10 多尺度训练的策略
作者在YOLOv2中采用了多尺度训练的策略，即在训练过程中随机选择不同的图片尺寸，以让网络能够在不同的输入尺寸上都能达到一个很好的预测效果。这种策略允许同一个网络在不同分辨率上进行检测，从而权衡速度和精度的表现。这是一种有效的方法，使得YOLOv2能够更加健壮地运行于不同尺寸的图片之上。

# 8. 目标检测常见指标:
**当有框出来的时候就confidence都会大于阈值**
1. True Positive TP: 
- 预测出来的label跟实际的label对上了
- Bounding Box 和 Ground Truth 的IOU大于阈值
2. False Positive: 负样本被检测为正样本(误报)
- 首先肯定是捡出来了
- label不匹配
- IOU低于阈值
3. False Negative: 错误的负向预测
- 正样本没被检测为负样本的数量，也称漏报，指没有检测出的 Ground Truth 区域
4. True Negative: 正确的负向预测
- 是负样本且被检测出的数量，无法计算，在目标检测中，通常也不关注 TN

Recall:表示被模型正确检测到的正样本占所有实际正样本的比例。在目标检测任务中，Recall 用于评估模型在检测到目标时的能力，即检测到的目标占所有实际目标的比例。
![在这里插入图片描述](https://img-blog.csdnimg.cn/9a6392ffe5714566af40d79e11902b37.png)


Precision:
![在这里插入图片描述](https://img-blog.csdnimg.cn/74eefab2bd7c41fbbd50a56893091e7d.png)


Mean Average Precision (mAP):
$$\text{mAP}=\frac{1}{\text{N}}\sum_{i=1}^{\text{N}}\text{AP}_i$$

- 其中，TP表示True Positive，FN表示False Negative，TN表示True Negative，FP表示False Positive，N表示类别数，AP表示Average Precision。


5. **PR曲线**：Precision-Recall曲线
6. AP：PR曲线下的面积，综合考量了 recall 和 precision 的影响，反映了模型对某个类别识别的好坏。
7. mAP：mean Average Precision, 即各类别AP的平均值，衡量的是在所有类别上的平均好坏程度。
8. 