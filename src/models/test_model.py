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

    # Model definition 
    model_type = GRU()
    model = Pl_module(model_type)
    model = model.load_from_checkpoint(model=model_type,checkpoint_path =args.infer_from_ckpt)
    model.to(device)
    # Run model on file
    test_model(args,model)

if __name__ == '__main__':
    main()

