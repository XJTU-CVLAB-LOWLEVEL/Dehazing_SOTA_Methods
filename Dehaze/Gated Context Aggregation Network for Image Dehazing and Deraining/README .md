Gated Context Aggregation Network for Image Dehazing and Deraining
=======
![image](imgs/net_arch.png)

This is the implementation of our WACV 2019 paper *"Gated Context Aggregation Network for Image Dehazing and Deraining"* by [Dongdong Chen](<http://www.dongdongchen.bid/>), [Mingming He](<https://github.com/hmmlillian>), [Qingnan Fan](<https://fqnchina.github.io/>), *et al.*

## Getting Started
## Model
Pre-trained model(epoch:44) can be downloaded from [baidu_disk](https://pan.baidu.com/s/1gsZqDvGFx1tAKtBnBAb_lg)(code:3fgb)

the model location:```/checkpoint/default_exp/gca_model.pt```

## Dataset prepare
1. Use our datasets.
2. Make dataset structure be:
- examples
    - train
	     - haze
		     - 100001.jpg
		     - 100002.jpg
	     - gt
		     - 100001.jpg
		     - 100002.jpg
    - test
	     - haze
		     - 100001.jpg
	     - gt
		     - 100001.jpg
## Train
```python train.py```

tags:You need to modify some params on certain lines in haze.py

	input_folder,  # train_haze_dir
	gt_folder # train_gt_dir
	test_input_folder # test_haze_dir
	test_gt_folder # test_gt_dir
	
## Test
```python test.py```



Cite
----

You can use our codes for research purpose only. And please cite our paper when you use our codes.
```
@article{chen2018gated,
  title={Gated Context Aggregation Network for Image Dehazing and Deraining},
  author={Chen, Dongdong and He, Mingming and Fan, Qingnan and Liao, Jing and Zhang, Liheng and Hou, Dongdong and Yuan, Lu and Hua, Gang},
  journal={WACV 2019},
  year={2018}
}
```


