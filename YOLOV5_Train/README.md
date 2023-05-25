
# yolov5的工程使用(以人员检测为案例)

使用ubuntu为案例
```bash
docker run --gpus all -it -p 6007:6006 -p 8889:8888 --name my_torch -v $(pwd):/app easonbob/my_torch1-pytorch:22.03-py3-yolov5-6.0
```
使用端口映射功能也就是说打开jupyter lab的指令是
```bash
http://localhost:8889/lab
```
当然，个人建议直接去vscode端口点击就打开jupyter lab和tensorboard比较方便

# 1. yolo数据格式
YOLO格式的标签文件是一个纯文本文件，每个文件名对应一张图像，每个标签文件中包含了该图像中所有检测到的目标的信息。

YOLOv5的标签格式包含了每个目标的类别和位置信息。具体来说，每个标签文件的每一行都包含了一个目标的信息，每个目标的信息由以下7个字段组成，用空格分隔：
```bash
<class> <x_center> <y_center> <width> <height> <confidence> <flag>
```
其中，<class>是目标的类别，是一个整数；<x_center>和<y_center>是目标的中心点相对于图像宽度和高度的比例；<width>和<height>是目标的宽度和高度相对于图像宽度和高度的比例；<confidence>是目标检测的置信度，用0到1之间的实数表示；<flag>是一个标志位，可以忽略。

例如，下面是一个YOLOv5格式的标签文件的示例：
```bash
0 0.456 0.678 0.123 0.234 0.9876
1 0.123 0.345 0.456 0.567 0.8765
```

# 2. 跑通人员检测(WiderPerson 数据集的案例的类别)

## 2.1 先看类别

```bash
# 一共是5类
0 pedestrians
1 riders
2 partially-visible persons
3 ignore regions
4 crowd
```


## 2.2 制作.yaml配置文件
**先看原版 coco128.yaml文件**
```bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: python train.py --data coco128.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here (7 MB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names


# Download script/URL (optional)
download: https://ultralytics.com/assets/coco128.zip
```

制作自己的person.yaml文件, 复制到data里面去

```bash
path: ../person_data  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# classes 
nc: 5
names: ['pedestrians', 'riders', 'partially-visible persons', 'ignore regions', 'crowd']
```

## 2.3 制作model.yaml

复制model/yolov5s.yaml文件，因为要修改类别， 改nc就行了, 简单对比下跟l, n, x， 深度和宽度不同，参数量也不同

```bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 5  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

## 2.4 使用以下命令就可以训练了

```bash
python ./train.py --data ./data/person.yaml --cfg ./models/yolov5s_person.yaml --weights ./weights/yolov5s.pt --batch-size 2 --epochs 1 --workers 0 --name s_121 --project yolo_person_s
```

**解释**
python: 运行命令的 Python 解释器。
./train.py: yolov5 提供的训练脚本，用于训练目标检测模型。
--data ./data/person.yaml: 数据集配置文件的路径，其中 person.yaml 是数据集的配置文件，里面包含了数据集的路径、类别等信息。
--cfg ./models/yolov5s_person.yaml: 模型配置文件的路径，其中 yolov5s_person.yaml 是 yolov5 基于 yolov5s 模型修改后的配置文件，用于适应特定的数据集和任务。
--weights ./weights/yolov5s.pt: 预训练模型权重文件的路径，其中 yolov5s.pt 是 yolov5 基于 yolov5s 模型在 COCO 数据集上预训练的权重文件。
--batch-size 2: 每个批次的图像数量，这里设置为 2。
--epochs 1: 训练的轮数，这里设置为 1，即只训练一轮。
--workers 0: 用于训练的进程数，这里设置为 0，表示不使用多进程加速训练，而是使用单进程进行训练。
--name s_121: 训练的名称，这里设置为 s_121，自动生成的文件夹在project下面,s是因为使用的是s模型，名字而已
--project yolo_person_demo: 训练项目的名称，这里设置为 yolo_person_demo。自动生成yolo_person_demo文件夹


## 2.5 增加合适的batch size
```watch nvidia-smi```这条指令可以每两秒钟查看一次显卡的显存使用率。调整合适的batch_size来满足对应的

主要是看**Memory-Usage**, 增加的倍数是16的倍数，A100是196好像是不记得了

```bash
Every 2.0s: nvidia-smi                                                     46f879adf741: Tue May 23 04:18:14 2023

Tue May 23 04:18:14 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A40-12Q      On   | 00000000:00:0B.0 Off |                    0 |
| N/A   N/A    P8    N/A /  N/A |      0MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


## 2.6 结果出炉

**这里只跑了一轮**

```bash
AutoAnchor: 4.93 anchors/target, 0.999 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to yolo_person_demo/s_1212
Starting training for 1 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       0/0    0.512G   0.07661    0.1915   0.02485        85       640: 100%|██████████|
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 
                 all       1000      28423      0.783      0.208      0.201      0.105

1 epochs completed in 0.236 hours.
Optimizer stripped from yolo_person_demo/s_1212/weights/last.pt, 14.4MB
Optimizer stripped from yolo_person_demo/s_1212/weights/best.pt, 14.4MB

Validating yolo_person_demo/s_1212/weights/best.pt...
Fusing layers... 
YOLOv5s_person summary: 213 layers, 7023610 parameters, 0 gradients, 15.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 
                 all       1000      28423      0.784      0.207      0.201      0.105
         pedestrians       1000      17833      0.545      0.827        0.8       0.46
              riders       1000        185          1          0    0.00518    0.00235
partially-visible persons       1000       9335      0.374      0.208      0.196     0.0584
      ignore regions       1000        409          1          0    0.00319     0.0012
               crowd       1000        661          1          0    0.00099   0.000297
Results saved to yolo_person_demo/s_1212
```

最后在yolov5的文件夹中生成了一个yolo_person_demo文件夹，下面是s1212文件夹下面存放着训练的结果。 这里我们设置的是s121, 这里是s1212，是因为第二次运行这个指令了，每次运行建议更改name这个参数

## 2.7 对训练结果进行可视化
这里使用训练好的结果
```
tensorboard --logdir=./yolo_person_demo
```

docker内部打开最简单办法, 点就完事了
![在这里插入图片描述](https://img-blog.csdnimg.cn/088bc3b7ac90480ca16312eb73bd930c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7ab0d491a67948519e89c4f29848e172.png)


## 2.8 模型的测试

使用**detect.py**脚本

```bash
python detect.py --source ./test.mp4 --weights ./yolo_person_demo/s_120/weights/best.pt --conf-thres 0.3
```

```bash
python detect.py --weights ./weights/yolov5s.pt --img 640 --conf 0.4 --source ./data/images/zidane.jpg --classes 0
```


## 2.9 模型的评估

使用**val.py**脚本

```bash
python val.py --data  ./data/person.yaml  --weights ./yolo_person_s/s_120/weights/best.pt --batch-size 12
```

python val.py --data  ./data/yolov5s_person.yaml  --weights ./weights/yolov5s.pt --batch-size 12


```bash
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 
                     all       1000      28027      0.451      0.372      0.375      0.209
             pedestrians       1000      17600      0.738      0.854      0.879      0.608
                  riders       1000        185      0.546      0.492      0.522      0.256
 artially-visible-person       1000       9198      0.461      0.334      0.336      0.125
          ignore-regions       1000        391       0.36      0.132      0.116     0.0463
                   crowd       1000        653      0.152     0.0468     0.0244    0.00841
```
**指标说明**

P: 准确率(Precision),检测出的正确目标数除以检测出的总目标数。

R: 召回率(Recall),检测出的正确目标数除以标注的总目标数。

mAP@.5: Mean Average Precision（平均精度均值）在 Intersection over Union（IoU）阈值为 0.5 时的值。mAP 是一种衡量目标检测模型性能的指标，它结合了 Precision 和 Recall。在计算 mAP 时，将 Precision-Recall 曲线下方的面积进行平均。数值越接近 1，表示模型性能越好。

mAP@.5:.95: 这是在 IoU 阈值从 0.5 到 0.95 之间以 0.05 为间隔的范围内计算的 Mean Average Precision。这是一种更严格的评估方法，因为它考虑了不同 IoU 阈值下的性能。数值越接近 1，表示模型性能越好。

Yolo在pedestrians(行人)类别上表现最好,mAP达到0.879。在riders(骑手)和partially-visible-person(部分可见人)上也还不错。

crowd(拥挤场景)和ignore-regions(忽略区域)的性能较差,因为目标比较小且密集难以检测。

mAP@.5较高,说明Yolo在低阈值下的检测性能较好。mAP@.5:.95较低,在高阈值下的性能有待提高。

P和R值都不算很高,说明Yolo的检测结果里面既包含遗漏的目标(R较低),也包含误检目标(P较低)。总体来说性能尚可,但有提高的空间。

# 3. 模型的导出(decode plugin)

使用plugin decode来加速yolov5的解码性能

**修改detect.py -> onnx -> simplify onnx -> 导出onnx**


## 3.1 decode plugin的使用

这里使用了一个 export.patch 的代码, 修改完后yolo.py 和 export.py这两个脚本会发生变化

```bash
git am export.patch
```

出现以下显示运行成功了
```bash
(base) root@d9f903dab148:/app/yolov5_used# git am export.patch
Applying: Enable onnx export with decode plugin
.git/rebase-apply/patch:108: trailing whitespace.
    
.git/rebase-apply/patch:186: trailing whitespace.
    
.git/rebase-apply/patch:205: trailing whitespace.
    
warning: 3 lines add whitespace errors.
```

安装导出onnx所需要的包
```bash
pip3 install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip3 install onnx-graphsurgeon -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip3 install onnx-simplifier==0.3.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```bash
apt update
apt install -y libgl1-mesa-glx
```

导出自己训练好的行人检测的权重
```bash
python export.py --weights yolo_person_demo/s_120/weights/best.pt --include onnx --simplify --dynamic
```

## 3.2 以yolov5s为例子对比decode plugin更改过的onnx
导出**yolov5s.onnx**, 这里使用的是改过的**export.py**
```bash 
python export.py --weights weights/yolov5s.pt --include onnx --simplify --dynamic
```

使用原版的**export.py**, 这个是直接从原版仓库里面拿的, 为了对比我去掉了--dynamic
```bash
python export_origin.py --weights weights/yolov5ss.pt --include onnx --simplify
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/939bfb943cde4f64b9cfcb1a4cc24288.png)



## 3.3 export.py的改变

**原版的export_onnx函数**
```bash
def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')
```

**更改过的export_onnx函数**
```bash
def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    # try:
    check_requirements(('onnx',))
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')
    print(train)
    torch.onnx.export(
        model,
        im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['p3', 'p4', 'p5'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,3,640,640)
            'p3': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,25200,4)
            'p4': {
                0: 'batch',
                2: 'height',
                3: 'width'},
            'p5': {
                0: 'batch',
                2: 'height',
                3: 'width'}
        } if dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    # Simplify
    if simplify:
        # try:
        check_requirements(('onnx-simplifier',))
        import onnxsim

        LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx,
                                                dynamic_input_shape=dynamic,
                                                input_shapes={'images': list(im.shape)} if dynamic else None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, f)
        # except Exception as e:
        #     LOGGER.info(f'{prefix} simplifier failure: {e}')

    # add yolov5_decoding:
    import onnx_graphsurgeon as onnx_gs
    import numpy as np
    yolo_graph = onnx_gs.import_onnx(model_onnx)
    p3 = yolo_graph.outputs[0]
    p4 = yolo_graph.outputs[1]
    p5 = yolo_graph.outputs[2]
    decode_out_0 = onnx_gs.Variable(
        "DecodeNumDetection",
        dtype=np.int32
    )
    decode_out_1 = onnx_gs.Variable(
        "DecodeDetectionBoxes",
        dtype=np.float32
    )
    decode_out_2 = onnx_gs.Variable(
        "DecodeDetectionScores",
        dtype=np.float32
    )
    decode_out_3 = onnx_gs.Variable(
        "DecodeDetectionClasses",
        dtype=np.int32
    )

    decode_attrs = dict()

    decode_attrs["max_stride"] = int(max(model.stride))
    decode_attrs["num_classes"] = model.model[-1].nc
    decode_attrs["anchors"] = [float(v) for v in [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]]
    decode_attrs["prenms_score_threshold"] = 0.25

    decode_plugin = onnx_gs.Node(
        op="YoloLayer_TRT",
        name="YoloLayer",
        inputs=[p3, p4, p5],
        outputs=[decode_out_0, decode_out_1, decode_out_2, decode_out_3],
        attrs=decode_attrs
    )

    yolo_graph.nodes.append(decode_plugin)
    yolo_graph.outputs = decode_plugin.outputs
    yolo_graph.cleanup().toposort()
    model_onnx = onnx_gs.export_onnx(yolo_graph)

    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, f)
    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f
    # except Exception as e:
    #     LOGGER.info(f'{prefix} export failure: {e}')
```

## 3.4 逐行解释export.py的代码
更改导出的名字，之前是导出output, 更改了output节点的名字, 变为p3, p4, p5, 这里对应的是YoloLayer_TRT上面的三个sigmoid节点

如果dynamic为True,对于输入的节点和输出的节点, 把维度0， 2， 3设定为可以调整的对象, 维度1是通道就不可调整了
```python
torch.onnx.export(
        model,
        im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['p3', 'p4', 'p5'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,3,640,640)
            'p3': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,25200,4)
            'p4': {
                0: 'batch',
                2: 'height',
                3: 'width'},
            'p5': {
                0: 'batch',
                2: 'height',
                3: 'width'}
        } if dynamic else None)
```

检查onnx是否符合规范
```python
# Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
```

简化onnx
```python
# Simplify
if simplify:
    # try:
    check_requirements(('onnx-simplifier',))
    import onnxsim

    LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
    model_onnx, check = onnxsim.simplify(model_onnx,
                                            dynamic_input_shape=dynamic,
                                            input_shapes={'images': list(im.shape)} if dynamic else None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, f)
    # except Exception as e:
    #     LOGGER.info(f'{prefix} simplifier failure: {e}')
```

这一步为后续的解码准备了必要的输入输出的变量, 用英伟达的onnx_graphsurgeon包, p3, p4, p5是输入, decode_out_0/1/2/3 是对应的输出。有了输入输出才去添加yoloLayer
```python
# add yolov5_decoding:
    import onnx_graphsurgeon as onnx_gs
    import numpy as np
    yolo_graph = onnx_gs.import_onnx(model_onnx)
    p3 = yolo_graph.outputs[0]
    p4 = yolo_graph.outputs[1]
    p5 = yolo_graph.outputs[2]
    decode_out_0 = onnx_gs.Variable(
        "DecodeNumDetection",
        dtype=np.int32
    )
    decode_out_1 = onnx_gs.Variable(
        "DecodeDetectionBoxes",
        dtype=np.float32
    )
    decode_out_2 = onnx_gs.Variable(
        "DecodeDetectionScores",
        dtype=np.float32
    )
    decode_out_3 = onnx_gs.Variable(
        "DecodeDetectionClasses",
        dtype=np.int32
    )
```

接下来设置解码过程的属性，并将解码层（YoloLayer）添加到yolo_graph中，最后导出修改后的 ONNX 模型。

通过model拿到max_stride, num_classes, anchors(锚框), 目标检测的分阈值, 设置为0.25最后全部储存到字典。这一步也是为了Yolo_layer
```python
decode_attrs = dict()

decode_attrs["max_stride"] = int(max(model.stride))
decode_attrs["num_classes"] = model.model[-1].nc
decode_attrs["anchors"] = [float(v) for v in [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]]
decode_attrs["prenms_score_threshold"] = 0.25
```

输入点有了, 输出点有了, attrs也有了，定义名称为decode_plugin的节点, 然后加入yolo_graph， 然后最后输出
```python
decode_plugin = onnx_gs.Node(
        op="YoloLayer_TRT",
        name="YoloLayer",
        inputs=[p3, p4, p5],
        outputs=[decode_out_0, decode_out_1, decode_out_2, decode_out_3],
        attrs=decode_attrs
    )

    yolo_graph.nodes.append(decode_plugin)
    yolo_graph.outputs = decode_plugin.outputs
    yolo_graph.cleanup().toposort()
    model_onnx = onnx_gs.export_onnx(yolo_graph)

    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, f)
    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f
    # except Exception as e:
    #     LOGGER.info(f'{prefix} export failure: {e}')
```

# 4. yolo.py文件的更改
![在这里插入图片描述](https://img-blog.csdnimg.cn/e8b38d0b98a347f0a8111d0c90a45247.jpeg)
这一坨全部不要了就保留sigmoid就可以了，然后就是直接硬编码t就是int32
```bash
diff --git a/models/yolo.py b/models/yolo.py
index 02660e6..c810745 100644
--- a/models/yolo.py
+++ b/models/yolo.py
@@ -55,29 +55,15 @@ class Detect(nn.Module):
         z = []  # inference output
         for i in range(self.nl):
             x[i] = self.m[i](x[i])  # conv
-            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
-            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
-
-            if not self.training:  # inference
-                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
-                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
-
-                y = x[i].sigmoid()
-                if self.inplace:
-                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
-                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
-                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
-                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
-                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
-                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
-                    y = torch.cat((xy, wh, conf), 4)
-                z.append(y.view(bs, -1, self.no))
-
-        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
+            y = x[i].sigmoid()
+            z.append(y)
+        return z
 
     def _make_grid(self, nx=20, ny=20, i=0):
         d = self.anchors[i].device
-        t = self.anchors[i].dtype
+        # t = self.anchors[i].dtype
+        # TODO(tylerz) hard-code data type to int
+        t = torch.int32
         shape = 1, self.na, ny, nx, 2  # grid shape
         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
         if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
-- 
2.36.0
```

# 5. 收尾打包好这个容器


```bash
# 打开镜像找到对应的CONTAINER ID
docker ps 

# 提交容器更改为一个新的镜像
docker commit contaier_id yolov5-6.0-onnx 

# 在easonbob的容器标记为yolov5-6.0-onnx 
docker tag yolov5-6.0-onnx easonbob/my_torch1-pytorch:yolov5-6.0-onnx

# 上传
docker push easonbob/my_torch1-pytorch:yolov5-6.0-onnx
```
好了，一个全新的就建立好了
![在这里插入图片描述](https://img-blog.csdnimg.cn/e5dc3e5bbc6f4e6289eca894e4526b00.png)
试着运行一下新搞好的容器
```bash
docker run --gpus all -it --name v5_onnx -v $(pwd):/app easonbob/my_torch1-pytorch:yolov5-6.0-onnx
```

经过测试也可以使用。这个镜像就可以用了，把之前的全部删掉就好了