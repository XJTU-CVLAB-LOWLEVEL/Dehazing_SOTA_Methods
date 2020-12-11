# Domain Adaptation for Image Dehazing


This repository contains code for reproducing the paper ["Domain Adaptation for Image Dehazing"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.pdf) 

## Requirements and dependencies
* python 3.8.5 
* pytorch = 1.6.0
* torchvision = 0.7.0

## Dataset prepare
1. Use our datasets.
2. Make dataset structure be:
- datasets
    - dehazing
        - train_syn
            - 1.jpg
            - 2.jpg
        - train_syn_depth
            - 1.jpg
            - 2.jpg
        - train_re
            - 1.jpg
            - 2.jpg
        - train_re_depth
            - 1.jpg
            - 2.jpg
        - test_syn
            - 1.jpg
            - 2.jpg
        - test_re
            - 1.jpg
            - 2.jpg

Synthetic hazy image and its gt need to be spliced together, and real hazy image needn't gt. Every pair of images of training set needs a depth map.

## Train
- Train CycleGAN 
```
python train.py --dataroot ./datasets/dehazing --name run_cyclegan --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8091 --which_model_netG resnet_9blocks --lambda_A 1 --lambda_B 1 --lambda_identity 0.1   --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2  --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model cyclegan
```

- Train Fr using the pretrained CycleGAN
```
python train.py  --dataroot ./datasets/dehazing --name run_fr_depth --lambda_Dehazing 10 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-2 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8090  --epoch_count 1 --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model RDehazingnet --g_s2r_premodel ./checkpoints/run_cyclegan/latest_netG_A.pth  
```

- Train Fs using the pretrained CycleGAN
```
python train.py  --dataroot ./datasets/dehazing --name run_fs_depth --lambda_Dehazing 10 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-2 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8094  --epoch_count 1 --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model SDehazingnet --g_r2s_premodel ./checkpoints/run_cyclegan/latest_netG_B.pth 
```

- Train DA_dehazing using the pretrained Fr, Fs and CycleGAN.
```
python train.py  --dataroot ./datasets/dehazing --name run_danet_depth --epoch_count 1 --niter 50 --lambda_S 1 --lambda_R 1 --lambda_identity 0.1 --lambda_Dehazing 10 --lambda_Dehazing_Con 0.1 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-3 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8094 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model danet --S_Dehazing_premodel ./checkpoints/run_fs_depth/latest_netS_Dehazing.pth --R_Dehazing_premodel ./checkpoints/run_fr_depth/latest_netR_Dehazing.pth --g_s2r_premodel ./checkpoints/run_cyclegan/latest_netG_A.pth --g_r2s_premodel ./checkpoints/run_cyclegan/latest_netG_B.pth --d_r_premodel ./checkpoints/run_cyclegan/latest_netD_A.pth --d_s_premodel ./checkpoints/run_cyclegan/latest_netD_B.pth
```


## Test

- Test Synthetic
```
python test.py --dataroot ./datasets/dehazing --name run_test --learn_residual --resize_or_crop crop --display_port 8095 --test_type syn --which_model_netG resnet_9blocks  --batchSize 1 --gpu_id 0 --model SDehazingnet --S_Dehazing_premodel ./checkpoints/run_danet_depth/latest_netS_Dehazing.pth
```

- Test Real
```
python test.py --dataroot ./datasets/dehazing --name run_test --learn_residual --resize_or_crop crop --display_port 8095 --test_type real --which_model_netG resnet_9blocks  --batchSize 1 --gpu_id 0 --model RDehazingnet --R_Dehazing_premodel ./checkpoints/run_danet_depth/latest_netR_Dehazing.pth
```
 
Test result will be stored in '/results/'
