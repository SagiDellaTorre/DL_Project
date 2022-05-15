from re import M
import torch
import warnings
import matplotlib.pyplot as plt
import soundfile
warnings.filterwarnings("ignore", category=UserWarning)
import torchaudio
# torchaudio.set_audio_backend("sox_io")
import librosa
import numpy as np
from scipy.io.wavfile import write
import csv
import pandas as pd
# ==================== Hyper-Params =================================================================================
epsilon = 1e-12

def create_input(args,mic,ref):
    pad_before = args.context - 1
    ref = torch.nn.functional.pad(ref, [0, 0, pad_before, 0])
    # ref = ref.pad(ref, [0, 0, pad_before, 0])
    ref = ref.unfold(dimension=0, size=args.context, step=1)
    input_data = torch.cat((mic.unsqueeze(2),ref),axis = 2)
    return input_data


class AudioPreProcessing():
    def __init__(self,args, device=None):
        self.args = args
        self.device = device

    def transformations(self,target_wav,feature_wav):
        # load signal

        freq_num = self.args.frequency_num

        with open(feature_wav, 'r', newline='') as f:

            line_read = csv.reader(f, delimiter=',')
            next(line_read) #first line is header
            next(line_read) #second line is header

            # loop over all the lines, convert the string to float, and append to the list
            for i, row in enumerate(line_read):
                row_data = row[1:] #first column is header
                feature = [float(i) for i in row_data]
                
                feature = torch.tensor(feature)
                feature = torch.reshape(feature, (1,freq_num,-1))
                if i == 0:
                    feature_mat = feature 
                else:
                    feature_mat = torch.cat((feature_mat, feature), 0)
    
        x = pd.read_csv(target_wav,index_col=False, header=None) 
        #convert to numpy and remove headers
        lables = x.values[1:,1:]

        target = torch.from_numpy(lables) # cast to torch tensor
        target = torch.tensor(target,dtype=torch.float) # cast to float
        input_data = feature_mat

        return target, input_data
