# 目录

<!-- TOC -->

- [目录](#目录)
- [RepLKNet描述](#RepLKNet描述描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [测试性能](#测试性能)

<!-- /TOC -->

# [RepLKNet描述](#目录)

RepLKNet模型是基于大卷积核创建的CNN模型，网络结构类似于ViT，使用大核卷积极大的增加量模型的感受野。

# [数据集](#目录)

使用的数据集：[caltech](https://xihe.mindspore.cn/datasets/drizzlezyk/caltech_for_user/tree)

- 数据集大小：共256个类、RGB彩色图像
    - 训练集：共20,270张图像
    - 测试集：共5,120张图像
- 数据格式：JPEG
    - 注：数据在caltech256.py中处理。
- 下载数据集，目录结构如下：

 ```text
├── train    # 训练集文件夹，每个类别的图片单独存放于一个文件夹下
│     ├─── 0    # 文件夹的名称代表该类别的label
│     │    ├─── 0.jpg   # label为0的训练图像
│     │    ├─── ...
│     ├─── ...
│     ├─── 256    # 文件夹的名称代表该类别的ID
│
├── test # 测试集文件夹
│     ├─── 0.jpg
│     ├─── ...
│     ├─── 1.jpg
│
├── result_example.txt # 提交的结果格式示例
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://mindspore.cn/tutorials/experts/zh-CN/r1.8/others/mixed_precision.html)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```bash
├── inference    # 在线推理相关代码目录
│  ├── app.py              # 推理核心启动文件
│  ├── requirements.txt    # 推理可视化相关依赖文件
│  ├── models
│  │    ├── Nets.py        # 网络结构文件 
│  │    ┕── Modules.py     # 网络结构依赖的模块 
│  └── example_img
│       ┕── airplane.jpg   # 示例图片 
│ 
├── train        # 在线训练相关代码目录
    ├── README.md            # 说明文档，说明每个文件及运行方式
    ├── config.json          # 训练配置文件，用于指定代码路径、超参数、数据路径等
    └── train_dir            # 训练代码所在的目录
        ├── pip-requirements.txt  # 训练代码所需要的package依赖声明文件
        ├── src
        │    ├──configs                          # RepLKNet的配置文件
        │    ├──data                             # 数据集配置文件
        │    │    ├──caltech256.py               # caltech256数据集读取文件
        │    │    ┕──ugment                      # 数据增强函数文件夹
        │    ├──models                           # 模型定义文件夹
        │    │    ┕──Nets                        # RepLKNet定义文件
        │    ├──trainers                         # 自定义TrainOneStep文件
        │    ├──tools                            # 工具文件夹
        │         ├──callback.py                 # 自定义回调函数，训练结束测试
        │         ├──cell.py                     # 一些关于cell的通用工具函数
        │         ├──criterion.py                # 关于损失函数的工具函数
        │         ├──get_misc.py                 # 一些其他的工具函数
        │         ├──optimizer.py                # 关于优化器和参数的函数
        │         ┕──schedulers.py               # 学习率衰减的工具函数
        ├── train.py                             # 训练文件
        ├── eval.py                              # 评估文件（由于数据集本身没有评估文件夹，需要在msnet.py中设置val_split=True才可运行）
        ├── predict.py                           # 预测文件（生成result.txt）
```

## 脚本参数

在msnet.yaml中可以同时配置训练参数和评估参数。

- 配置RepLKNet和caltech256数据集。

  ```python
  # Architecture
  arch: MSNet
  
  # ===== Dataset ===== #
  set: Caltech256
  num_classes: 256
  mix_up: 0.8
  cutmix: 1.0
  auto_augment: rand-m9-mstd0.5-inc1
  interpolation: bicubic
  mixup_prob: 1.
  switch_prob: 0.5
  mixup_mode: batch
  crop: True
  re_prob: 0.25
  re_mode: pixel
  re_count: 1
  val_split: False  # 自动分割数据集为训练集＆验证集
  test_name: "test-final"  # 测试集的文件夹名称
  
  # ===== Learning Rate Policy ======== #
  optimizer: momentum
  base_lr: 0.01
  warmup_lr: 0.000007
  min_lr: 0.00005
  lr_scheduler: cosine_lr
  warmup_length: 10
  
  # ===== Network training config ===== #
  amp_level: O1
  keep_bn_fp32: True
  beta: [ 0.9, 0.999 ]
  clip_global_norm_value: 1.
  is_dynamic_loss_scale: True
  epochs: 100
  label_smoothing: 0.1
  weight_decay: 0.0004
  momentum: 0.9
  batch_size: 32
  
  # ===== Hardware setup ===== #
  num_parallel_workers: 16
  device_target: Ascend
  device_num: 1
  graph_mode: 0
  
  # ===== Model config ===== #
  drop_path_rate: 0.5
  drop_rate: 0.0
  image_size: 224
  ```

更多配置细节请参考脚本`msnet.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动训练
  python train.py --pretain_path HenonBamboo/replknet_pre/weight_xl.ckpt --train_data_path drizzlezyk/caltech_for_user --output_path ./ckpt_0/best_model.ckpt

  # 使用Python启动验证。前提：需要将msnet.yaml中的val_split设置为True。
  python eval.py --model_path HenonBamboo/ai_classifier/best_model.ckpt --test_data_path drizzlezyk/caltech_for_user

  # 使用Python启动预测
  python predict.py --model_path HenonBamboo/ai_classifier/best_model.ckpt --test_data_path drizzlezyk/caltech_for_user --output_path ./ckpt_0/result.txt
  ```
  

# [模型描述](#目录)

## 测试性能

#### Caltech256上的RepLKNet

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型|RepLKNet|
| 模型版本              | RepLKNet-31XL                                                |
| 资源                   | Ascend 910               |
| MindSpore版本          | 1.7.0                                               |
| 数据集                    | Caltech256 Train，共20,270张图像                                              |
| 训练参数        | epoch=100, batch_size=32            |
| 优化器                  | momentum                                                    |
| 损失函数              | SoftTargetCrossEntropy                                       |
| 输出                    | 概率                                                 |
| 分类准确率             | 单卡：top1:97.0%                 |
