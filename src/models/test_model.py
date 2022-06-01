from ast import arg
import sys
sys.path.append('src/')

import numpy as np
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from scipy.io.wavfile import write
import soundfile
import librosa
from def_model import *
from collections import OrderedDict
import glob
import os
from data.metrics import test_model
from pathlib import Path
ROOT_PATH = Path(__file__).parent.parent.parent
epsilon = 1e-6

def create_dirs(dirs_list):
    """
    create the directories, if they aren't exist
    """
    for directory in dirs_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

class Pl_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        pred = self.model(x)
        return pred
        
# ======================================== main section ==================================================

Hydra_path = str(ROOT_PATH) + "/src/conf"
@hydra.main(config_path=Hydra_path,config_name="test.yaml")
def main(args):

    path = str(ROOT_PATH)
    
    # Model definition 
    model_type = GRU()
    model = Pl_module(model_type)
    
    for ckpt_file in args.infer_from_ckpt:
    
        # directory naming
        ckpt_file_path = path + '/' + ckpt_file
        names_list = ckpt_file.split('/')
        model_name = names_list[names_list.index("models") + 1]
        version_num = names_list[names_list.index("models") + 3]
        ckpt_name = names_list[-1].split('.')[0]

        create_dirs(
            [str(ROOT_PATH) + '/' + args.reports_directory + 'figures/' + model_name,
            str(ROOT_PATH) + '/' + args.reports_directory + 'figures/' + model_name + '/' + version_num,
            str(ROOT_PATH) + '/' + args.reports_directory + 'figures/' + model_name + '/' + version_num + '/' + ckpt_name]
        )
        report_dir = str(ROOT_PATH) + '/' + args.reports_directory + 'figures/' + model_name + '/' + version_num + '/' + ckpt_name + '/'

        model = model.load_from_checkpoint(model=model_type, checkpoint_path = ckpt_file_path)
        model.to(device)
        
        # Run model on file
        test_model(args, model, report_dir)

if __name__ == '__main__':
    main()

