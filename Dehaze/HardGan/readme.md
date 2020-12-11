# HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing

## News

[2020/11/05] Our code released.

## Reuqirements

- Python == 3.7.0  
- [Pytorch](https://pytorch.org/) == 1.6.0
- Torchvision == 0.7.0  
- Scipy == 1.1.0


## Quick start

+ `mkdir checkpoint data NH_results training_log`
+ Download datasets into ./data folder
+ Use `bash run.sh`
+ If you train on REDIES datasets, please use train_phrase 1.
+ If you train on NTIRE2020， please train train_phrase 1 first. Then, copy trained trained parameter to G1 and G2(crop_size is [240, 240]).  Finetune G2.  Train G3 finally(crop_size is [960, 960]). You can choose whether to do data augmentation by aug_fog.py.

## Model
Pre-trained model(epoch:200) can be downloaded from [baidu_disk](https://pan.baidu.com/s/1iNSzTkoHIel5riyf8KEirQ)(code:yrb6)
the model location:```/checkpoint/1_0.tar```

## Dataset prepare
1. Make dataset structure be:
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


1.```python train_nitre.py```

tags:You need to modify some params on certain lines in train_nitre.py  &&  train_nitre_data.py

	train_data_dir # train_dir
	train_list # train_data_list
	val_data_dir # test_dir
	val_list # test_data_list
	
## Test
1.```python test_nitre.py```

## Citation

If your find our research is helpful for you, please cite our paper.


