
本代码库基于检测代码库[detectron2](https://github.com/facebookresearch/detectron2)和蒸馏代码库[RepDistiller](https://github.com/HobbitLong/RepDistiller)，完成将蒸馏方法应用在faster-rcnn的代码库。

## 1. 教师模型下载

首先您需要下载教师模型，可参考[下载地址](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)，并放置在如下地址:
```
--Dilation-FasterRCNN
    --detectron2
        --teacher_models
```
修改configs中文件的相关参数，将DISTILL.ENABLE设置为True则在您的工作中加入蒸馏

## 2. 运行代码示例
单GPU运行
```
python tools/train_net.py   --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```
多GPU运行
```
python tools/train_net.py   --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

## 3. 模型效果
以下是对FasterRCNN进行蒸馏/不进行蒸馏的模型对比：
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">distill or not</th>
<th valign="bottom">teacher model</th>
<th valign="bottom">distill loss</th>
<th valign="bottom">box AP</th>
<!-- TABLE BODY -->
<!-- ROW: FasterRCNN_R50-FPN -->
 <tr><td align="left">R50-FPN</a></td>
<td align="center">No</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">38.0</td>
</tr>
<!-- ROW: FasterRCNN_S_R50-FPN_T_R101-FPN_HintLoss -->
 <tr><td align="left">R50-FPN</a></td>
 <td align="center">Yes</td>
<td align="center">R101-FPN</td>
<td align="center">HintLoss</td>
<td align="center">40.0</td>
</tr>
</tbody></table>