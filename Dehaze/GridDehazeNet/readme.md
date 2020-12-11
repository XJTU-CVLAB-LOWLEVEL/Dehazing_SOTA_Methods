# GridDehazeNet
This repo contains the official training and testing codes for our paper:

### GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
[Xiaohong Liu](https://xiaohongliu.ca)<sup>[*](#myfootnote1)</sup>, Yongrui Ma<sup>[*](#myfootnote1)</sup>, Zhihao Shi, [Jun Chen](http://www.ece.mcmaster.ca/~junchen/)

<a name="myfootnote1">*</a> _Equal contribution_

Published on _2019 IEEE International Conference on Computer Vision (ICCV)_

[[Paper](https://proteus1991.github.io/GridDehazeNet/resource/GridDehazeNet.pdf)] [[Project Page](https://proteus1991.github.io/GridDehazeNet/)]
___


## Requirements and dependencies
- Python == 3.7.0  
- [Pytorch](https://pytorch.org/) == 1.6.0
- Torchvision == 0.7.0  
- Pillow == 5.2.0  
- Numpy == 1.16.0
- Scipy == 1.1.0


## Model
Pre-trained model(epoch:200) can be downloaded from [baidu_disk](https://pan.baidu.com/s/1iNSzTkoHIel5riyf8KEirQ)(code:yrb6)

the model location:```/indoor_haze_best_3_6```

## Dataset prepare
1. Use our datasets(Despite the name is RESIDE, the actual data set is our own).
2. Make dataset structure be:
- data
    - VideoHazy_v3
        - train
	        - hazy
		        - 100001.JPG
		        - 100002.JPG
	        - gt
		        - 100001.JPG
		        - 100002.JPG
		    - train_list.txt
        - test
	        - hazy
		        - 200001.JPG
	        - gt
		        - 200001.JPG
		    - test_list.txt

## Train


1.```python train.py```

tags:You need to modify some params on certain lines in train.py  &&  train_data.py

	train_data_dir # train_dir
	train_list # train_data_list
	val_data_dir # test_dir
	val_list # test_data_list
	
## Test
1.```python test.py```



## Citation
```
@inproceedings{DuRN_arxiv,
title={Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration},
author={Liu, Xing and Suganuma, Masanori and Sun, Zhun and Okatani, Takayuki},
booktitle={arXiv preprint arXiv:1903.08817},
year={2019},
}

@inproceedings{DuRN_cvpr19,
title={Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration},
author={Liu, Xing and Suganuma, Masanori and Sun, Zhun and Okatani, Takayuki},
booktitle={Proc. Conference on Computer Vision and Pattern Recognition},
pages={7007-7016},
year={2019},
}

```

