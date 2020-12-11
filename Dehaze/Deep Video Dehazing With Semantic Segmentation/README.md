# Deep Video Dehazing With Semantic Segmentation


## Requrirements and dependents

*   python = 3.7.0
*   pytorch = 1.6.0


## Model
  

Models:

*   model/VDH/VDH.pkl


## Dataset prepare
Make dataset structure be:  

datasets  

    VideoHazy_v2_synt  
        Test
            dx_Res_new
                C005
                    1
                        00000.JPG
                        00001.JPG
            dx_Transmission
                C005
                    1
                        00000.JPG
                        00001.JPG
            segment
                C005
                    1
                        00000.JPG
                        00001.JPG
        Train
            dx_Res_new
                C002_1
                    1
                        00000.JPG
                        00001.JPG
            dx_Transmission
                C002_1
                    1
                        00000.JPG
                        00001.JPG
            segment
                C002_1
                    1
                        00000.JPG
                        00001.JPG


## train
 python train.py

## test
python test1.py


## citation

1.  **Deep Video Dehazing With Semantic Segmentations**<br />
    Wenqi Ren, Jingang Zhang,Xiangyu Xu, Lin Ma). <br />
    [[link]](https://ieeexplore.ieee.org/document/8492451). In TIP, 2018.
