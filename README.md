This is the official implementation of our work End-to-End Semi-Supervised Learning for Video Action Detection at CVPR'22. [Paper](https://arxiv.org/abs/2203.04251)

![framework](https://user-images.githubusercontent.com/22872200/179375948-a81c8997-b60a-40d7-ad66-d54bbe2cb3ab.png)

## Train instructions

This is the command line argument to run the code respectively for variance and gradient maps:

```
python main.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4\
 --pkl_file_label train_annots_20_labeled.pkl\
 --pkl_file_unlabel train_annots_80_unlabeled.pkl\
 --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
 --const_loss l2\
 --bv --n_frames 5 --thresh_epoch 11\
 --exp_id cyclic_variance_maps
```

```
python main.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4\
 --pkl_file_label train_annots_20_labeled.pkl\
 --pkl_file_unlabel train_annots_80_unlabeled.pkl\
 --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
 --const_loss l2\
 --gv\
 --exp_id gradient_maps
```

Parameters explanation:
- *bv* - Temporal Variance Attentive Mask 
- *gv* - Gradient Smoothness Attentive Mask
- *wt_loc* - Weight for localization loss 
- *wt_cls* - Weight for classification loss 
- *wt_cons* - Weight for consistency loss
- *exp_id* -  Experiment id to set the folder name for saving checkpoints
- *pkl_file_label* - Labeled subset
- *pkl_file_unlabel* - Unlabeled subset

## Pre-trained weights

Link to download I3D pre-trained weights:  
```
https://github.com/piergiaj/pytorch-i3d/tree/master/models
```
We have used **rgb_charades.pt** for our experiments.

## Datasets

UCF101-24 splits: [Pickle files](https://drive.google.com/drive/u/0/folders/1aFlPKtzWIufyAOkcAmUySH4PB_uCPDkj)

JHMDB-21  splits: [Text files](https://drive.google.com/drive/u/0/folders/1whGR2pg299D5W7jDV9Rop_jpr1ENIALF)

Set data path for UCF101 videos in ucf_dataloader.py inside datasets.

## Results
![main results](https://user-images.githubusercontent.com/22872200/179379251-885932a9-6c32-4dd0-8ce7-eaf3dbd15f6e.png)


## Citation
If you find this work useful, please consider citing the following paper:

```
@InProceedings{Kumar_2022_CVPR,
    author    = {Kumar, Akash and Rawat, Yogesh Singh},
    title     = {End-to-End Semi-Supervised Learning for Video Action Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14700-14710}
}
```
