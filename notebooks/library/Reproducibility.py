import random
import torch
import os
import numpy as np


class Reproducibility():

    RANDOM_SEED = 1234


    def seed_everything(seed=None):
        """
        Thanks to 
        https://www.kaggle.com/kunwar31/simple-lstm-with-identity-parameters-fastai/log
        https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
        
        """
        if seed is None:
            seed = Reproducibility.RANDOM_SEED   

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
