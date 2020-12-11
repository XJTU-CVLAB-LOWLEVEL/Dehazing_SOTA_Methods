# Distilling Image Dehazing With Heterogeneous Task Imitation  


This repository contains code for reproducing the paper ["Distilling Image Dehazing With Heterogeneous Task Imitation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hong_Distilling_Image_Dehazing_With_Heterogeneous_Task_Imitation_CVPR_2020_paper.pdf)


## Requirements and dependencies
* python 3.8.5 
* pytorch = 1.6.0
* torchvision = 0.7.0


## Dataset prepare
1. Use our datasets.
2. Make dataset structure be:
- dataset
    - ITS_v2
        - hazy
            - 1_1.png
            - 1_2.png
            - 2_1.png
            - 2_2.png
        - gt
            - 1.png
            - 2.png
    - SOTS
        - hazy
            - 1400_1.png
            - 1400_2.png
            - 1401_1.png
            - 1401_2.png
        - gt
            - 1400.png
            - 1401.png
            
## Train and Test
- Teacher Training

1.```cd KDDN```

2.```python train_teacher.py --epochs 30```

tags:You need to modify some params on certain lines in train_teacher.py

	path_train # train_dir
	path_val # val_dir
	path_test # test_dir
	
Model and test result will be stored in 'aux/Results/teacher' and 'aux/CheckPoints/teacher'
	
- Student Training

1.```cd KDDN```

2.```python train.py --epochs 200 --load-best-teacher-model --best-teacher-model-path <model_path.pth>```

tags:You need to modify some params on certain lines in train.py

	path_train # train_dir
	path_val # val_dir
	path_test # test_dir
	
Model and test result will be stored in 'aux/Results/teacher+student' and 'aux/CheckPoints/teacher+student'