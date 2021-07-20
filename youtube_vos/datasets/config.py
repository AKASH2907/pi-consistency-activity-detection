from torch import cuda
from datetime import datetime

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

# n_epochs = 26
# batch_size = 4
n_frames = 8
skew_weight = False
n_anns = 7

# learning_rate = 5e-4
# weight_decay = 1e-7

# model_id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# save_dir = './save_models/Run_%s/' % model_id

# resume_model_path = './save_models/Run_2021-05-24 23,00,32/model_10_0.0972.pth'
# model_path = './save_models/Run_2021-05-24 23,00,32/model_10_0.0972.pth'

bce_w_logits = True
use_hidden_state = False
use_fixes = False
