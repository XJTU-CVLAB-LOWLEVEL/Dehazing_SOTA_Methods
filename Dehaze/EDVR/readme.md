
#### EDVR has been merged into [BasicSR](https://github.com/xinntao/BasicSR). This GitHub repo is a mirror of [BasicSR](https://github.com/xinntao/BasicSR). Recommend to use [BasicSR](https://github.com/xinntao/BasicSR), and open issues, pull requests, etc in [BasicSR](https://github.com/xinntao/BasicSR).
Note that this version is not compatible with previous versions. If you want to use previous ones, please refer to the `old_version` branch.

---


## Requirements and dependencies
- Python == 3.7.0  
- [Pytorch](https://pytorch.org/) == 1.6.0
- Torchvision == 0.7.0  
- Pillow == 5.2.0  
- Numpy == 1.16.0
- Scipy == 1.1.0
- addict
- future
- lmdb
- matplotlib
- mmcv>=0.6
- numpy
- opencv-python
- pyyaml
- scikit-image
- tb-nightly
- yapf

#### If you use the EDVR model, the above cuda extensions are necessary.
  ```python setup.py develop```


## Model
Pre-trained model can be downloaded from [baidu_disk](https://pan.baidu.com/s/1iNSzTkoHIel5riyf8KEirQ)(code:yrb6)

the model location:```/experiments/EDVR/models/net_g_10000.pth```
# 配置文件
## 实验命名

以`EDVR`为例:


## 配置文件说明

我们使用了 yaml 格式来做配置文件.

### 训练配置文件

我们以[train_MSRResNet_x4.yml]为例, 说明训练配置文件的含义:

```yml
#################
# 以下为通用的设置
#################
# 实验名称, 具体可参见 [实验名称命名], 若实验名字中有debug字样, 则会进入debug模式
name: EDVR
# 使用的model类型, 一般为在`models`目录下定义的模型的类名
model_type: SRModel
# 输出相比输入的放大比率, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
scale: 4
# 训练卡数
num_gpu: 1  # set num_gpu: 0 for cpu mode
# 随机种子设定
manual_seed: 0

#################################
# 以下为dataset和data loader的设置
#################################
datasets:
  # 训练数据集的设置
  train:
    # 数据集的名称
    name: DIV2K
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # 文件名字模板, 一般LQ文件会有类似`_x4`这样的文件后缀, 这个就是来处理GT和LQ文件后缀不匹配的问题的
    filename_tmpl: '{}'
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

    # 训练中Ground-Truth的Training patch的大小
    gt_size: 128
    # 是否使用horizontal flip, 这里的flip特指 horizontal flip
    use_flip: true
    # 是否使用rotation, 这里指的是每隔90°旋转
    use_rot: true

    #### 下面是data loader的设置
    # data loader是否使用shuffle
    use_shuffle: true
    # 每一个GPU的data loader读取进程数目
    num_worker_per_gpu: 6
    # 总共的训练batch size
    batch_size_per_gpu: 16
    # 扩大dataset的倍率. 比如数据集有15张图, 则会重复这些图片100次, 这样一个epoch下来, 能够读取1500张图
    # (事实上是重复读的). 它经常用来加速data loader, 因为在有的机器上, 一个epoch结束, 会重启进程, 往往会很慢
    dataset_enlarge_ratio: 100

  # validation 数据集的设置
  val:
    # 数据集名称
    name: Set5
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/Set5/LRbicx4
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

#####################
# 以下为网络结构的设置
#####################
# 网络g的设置
network_g:
  # 网络结构 (Architecture)的类型, 一般为在`models/archs`目录下定义的dataset的类名
  type: MSRResNet
  #### 以下属性是灵活的, 可以在相应类的说明文档中获得
  # 输入通道数目
  num_in_ch: 3
  # 输出通道数目
  num_out_ch: 3
  # 中间特征通道数目
  num_feat: 64
  # 使用block的数目
  num_block: 16
  # SR的放大倍数
  upscale: 4

######################################
# 以下为路径和与训练模型、重启训练的设置
######################################
path:
  # 预训练模型的路径, 需要以pth结尾的模型
  pretrain_model_g: ~
  # 加载预训练模型的时候, 是否需要网络参数的名称严格对应
  strict_load: true
  # 重启训练的状态路径, 一般在`experiments/exp_name/training_states`目录下
  # 这个设置了, 会覆盖  pretrain_model_g 的设定
  resume_state: ~


#################
# 以下为训练的设置
#################
train:
  # 优化器设置
  optim_g:
    # 优化器类型
    type: Adam
    ##### 以下属性是灵活的, 根据不同优化器有不同的设置
    # 学习率
    lr: !!float 2e-4
    weight_decay: 0
    # Adam优化器的 beta1 和 beta2
    betas: [0.9, 0.99]

  # 学习率的设定
  scheduler:
    # 学习率Scheduler的类型
    type: CosineAnnealingRestartLR
    #### 以下属性是灵活的, 根据学习率Scheduler有不同的设置
    # Cosine Annealing的周期
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing每次Restart的权重
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing的学习率最小值
    eta_min: !!float 1e-7

  # 总共的训练迭代次数
  total_iter: 1000000
  # warm up的iteration数目, 如是-1, 表示没有warm up
  warmup_iter: -1  # no warm up

  #### 以下是loss的设置
  # pixel-wise loss的options
  pixel_opt:
    # loss类型, 一般为在`basicsr/models/losses`目录下定义的loss类名
    type: L1Loss
    # loss 权重
    loss_weight: 1.0
    # loss reduction方式
    reduction: mean


#######################
# 以下为Validation的设置
#######################
val:
  # validation的频率, 每隔 5000 iterations 做一次validation
  val_freq: !!float 5e3
  # 是否需要在validation的时候保存图片
  save_img: false

  # Validation时候使用的metric
  metrics:
    # metric的名字, 这个名字可以是任意的
    psnr:
      # metric的类型, 一般为在`basicsr/metrics`目录下定义的metric函数名
      type: calculate_psnr
      #### 以下属性是灵活的, 根据metric有不同的设置
      # 计算metric时, 是否需要crop border
      crop_border: 4
      # 是否转成在Y(CbCr)空间上计算metric
      test_y_channel: false

####################
# 以下为Logging的设置
####################
logger:
  # 屏幕上打印的logger频率
  print_freq: 100
  # 保存checkpoint的频率
  save_checkpoint_freq: !!float 5e3
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~

#############################################################
# 以下为distributed training的设置, 目前只有在Slurm训练下才需要
#############################################################
dist_params:
  backend: nccl
  port: 29500
```



## Dataset prepare
1. Use our datasets(Despite the name is RESIDE, the actual data set is our own).
2. Make dataset structure be:
- data
    - VideoHazy_v3
        - train
			- hazy
				- clip1
					- 00000.JPG
					- 00001.JPG
					- ....
				- clip2
					- 00000.JPG
					- 000001.JPG
					- ...
			- gt
				- clip1
					- 00000.JPG
					- 00001.JPG
					- ....
				- clip2
					- 00000.JPG
					- 00001.JPG
					- .....
        - test
			- hazy
				- clip3
					- 00000.JPG
					- 00001.JPG
					- ....
				- clip4
					- 00000.JPG
					- 00001.JPG
					- ...
			- gt
				- clip3
					- 00000.JPG
					- 00001.JPG
					- ....
				- clip4
					- 00000.JPG
					- 00001.JPG
					- .....
- 1.Directly read disk data.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: /datasets/VideoHazy_v3/Train/gt
    dataroot_lq: /datasets/VideoHazy_v3/Train/hazy
    io_backend:
      type: disk
    ```
- 2.Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```
**meta information**

`train.txt`, We use txt file to record for readability. The contents are:

```txt
C002_1 21 (1800,1806,3)
C002_2 21 (1800,1806,3)
...
```


## Train&&Test



## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt /options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

### Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt /options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt /options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml


## Citation
```
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {BasicSR},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2020}
}
```

