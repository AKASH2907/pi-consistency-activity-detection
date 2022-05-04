This is the official implementation of our work End-to-End Semi-Supervised Learning for Video Action Detection at CVPR'22.
[![arXiv](https://img.shields.io/badge/arXiv-2203.04251-b31b1b.svg)](https://arxiv.org/abs/2203.04251)

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

This code runs for UCF101. For JHMDB, everything is similar except the dataloading, since there it's a localization mask compared to bounding boxes in UCF101.

Set data path for UCF101 videos in ucf_dataloader.py inside datasets.

## Citation
If you find this work useful, please consider citing the following paper:

```
@article{Kumar2022EndtoEndSL,
  title={End-to-End Semi-Supervised Learning for Video Action Detection},
  author={Akash Kumar and Yogesh Singh Rawat},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.04251}
}
```


Updates: I'll update readme file, jhmdb loader and dataset splits we used.
