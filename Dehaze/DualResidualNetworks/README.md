# Dual Residual Networks  
By [Xing Liu](https://scholar.google.com/citations?user=bdVU63IAAAAJ&hl=en)<sup>1</sup>, [Masanori Suganuma](https://scholar.google.co.jp/citations?user=NpWGfwgAAAAJ&hl=ja)<sup>1,2</sup>, [Zhun Sun](https://scholar.google.co.jp/citations?user=Y-3iZ9EAAAAJ&hl=en)<sup>2</sup>, [Takayuki Okatani](https://scholar.google.com/citations?user=gn780jcAAAAJ&hl=en)<sup>1,2</sup>


Tohoku University<sup>1</sup>, RIKEN Center for AIP<sup>2</sup>

[link to the paper](https://arxiv.org/pdf/1903.08817.pdf)

## Requirements and dependencies
* python 3.7 (recommend to use [Anaconda](https://www.anaconda.com/))
* pytorch = 1.6.0
* torchvision = 0.8.1

## Model
Pre-trained model(epoch:200) can be downloaded from [baidu_disk](https://pan.baidu.com/s/1iNSzTkoHIel5riyf8KEirQ)(code:yrb6)

the model location:```/train/trainedmodels/RESIDE/DURN_US/DURN.pt```

## Dataset prepare
1. Use our datasets(Despite the name is RESIDE, the actual data set is our own).
2. Make dataset structure be:
- data
    - RESIDE
        - indoor_train
	        - images
		        - 100001.jpg
		        - 100002.jpg
	        - labels
		        - 100001.jpg
		        - 100002.jpg
        - sots_indoor_test
	        - images
		        - 200001.jpg
	        - labels
		        - 200001.jpg
## Train
1.```cd train```

2.```python haze.py```

tags:You need to modify some params on certain lines in haze.py

	data_root # train_dir
	imlist_pth # train_data_list
	test_root # test_dir
	testlist_pth # test_data_list
	
## Test
1.```cd test```

2.```python haze.py```



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