
本代码库基于检测代码库[detectron2](https://github.com/facebookresearch/detectron2)和蒸馏代码库[RepDistiller](https://github.com/HobbitLong/RepDistiller)，完成将蒸馏方法应用在faster-rcnn的代码库。

## 1. 教师模型下载

首先您需要下载教师模型，可参考[下载地址](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)，并放置在如下地址:
```
--Dilation-FasterRCNN
    --detectron2
        --teacher_models
```
修改configs中文件的相关参数，将DISTILL.DO设置为True则在您的工作中加入蒸馏

## 2. 运行代码示例
单GPU运行
```
python tools/train_net.py   --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```
多GPU运行
```
python tools/train_net.py   --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```
## 3. 引用

如果您在研究中使用 Dilation-FasterRCNN ，请使用以下 BibTeX 条目。

```BibTeX
@misc{Dilation-FasterRCNN,
  author =       {JingyaoLi}},
  title =        {Dilation-FasterRCNN},
  howpublished = {\url{https://github.com/lijingyao20010602/Detectron2}},
  year =         {2021}
}
```