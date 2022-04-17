Hi, I apologize for delay in releasing codes. Please wait till **end of April**. Really very busy with coursework and projects. Thanks for understanding.

This is the official implementation of our work End-to-End Semi-Supervised Learning for Video Action Detection at CVPR'22.

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


Parameter *bv* and *gv* enables the variance and gradient mode calculations. Weights for labeled localization and classification loss is set by *wt_loc* and *wt_cls*. Similarly the last argument in that line is to set consistency loss weight. const_loss denotes the cosistency loss used. In our work, it's L2 loss. Experiment id to set the folder name for saving checkpoints.

This code runs for UCF101. For JHMDB, everything is similar except the dataloading, since there it's a localization mask compared to bounding boxes in UCF101.

Set data path for UCF101 videos in ucf_dataloader.py inside datasets.

Updates: I'll update readme file, jhmdb loader and dataset splits we used.
