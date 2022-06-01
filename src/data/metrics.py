import numpy as np
import os
import torch
from scipy.io.wavfile import write
import sys
import soundfile
import matplotlib.pyplot as plt
from data.data_utils import AudioPreProcessing
from pathlib import Path
ROOT_PATH = Path(__file__).parent.parent.parent

def create_dirs(dirs_list):
    """
    create the directories, if they aren't exist
    """
    for directory in dirs_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def inference(args, audio_pre_process, model, target_wav, feature_wav, report_dir, file_name):

    #device = torch.device('cuda:{:d}'.format(next(model.parameters()).device.index)) if torch.cuda.is_available() else 'cpu'

    target, input_data = audio_pre_process.transformations(target_wav,feature_wav)

    if torch.cuda.is_available():
        input_data = input_data.to(torch.device('cuda'))

    pred = model(torch.unsqueeze(input_data, 0))


    #Visualize spectogram
    fig, axs = plt.subplots(2, 1, constrained_layout=True)   
    fig.suptitle('NN predictions') 

    axs[0].plot(range(target.shape[0]), target[:,1], label="target")
    axs[0].plot(range(target.shape[0]), pred[0,:,1], label="prediction")
    axs[0].set_title('VAD figure')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('VAD')
    axs[0].legend()

    axs[1].plot(range(target.shape[0]), target[:,0], label="target")
    axs[1].plot(range(target.shape[0]), pred[0,:,0]*360, label="prediction")
    axs[1].plot(range(target.shape[0]), target[:,1]*100, label="VAD")
    axs[1].set_title('Angle figure')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Angle')
    axs[1].legend()

    plt.savefig(report_dir + file_name + '.png')

    return 

def pred_model(args,model,output_name,save_output_files):
    
    model = model.eval()
    # filelist_mic = os.listdir(args.path_mic_wav)
    # filelist_ref = os.listdir(args.path_ref_wav)
    # filelist_speech = os.listdir(args.path_speech_wav)

    filelist = os.listdir(args.path_wav)
    filelist_mic = list()
    for file in filelist:
        if file.endswith('mic.wav'):
            filelist_mic.append(file)
    
    audio_pre_process = AudioPreProcessing(args)

    with torch.no_grad():
        for x in range(len(filelist_mic)):
            #print(x/len(filelist_mic))
            tmp = filelist_mic[x]
            mic_wav = args.path_wav + tmp
            ref_wav = args.path_wav + tmp[0:-8] + '_lpb.wav'

            output = inference(args,audio_pre_process,model,mic_wav,ref_wav)
            

            # save outputs
            if save_output_files:
                tmp=filelist_mic[x]
                output_file_generated = args.path_wav + tmp[0:-8] + output_name
                write(output_file_generated, args.sample_rate, (output*2**15).cpu().numpy().T.astype(np.int16)) 

                #output_file_generated = args.path_wav + tmp[0:-8] + '_output_regular_rls.wav'
                #write(output_file_generated, args.sample_rate, (output_rls*2**15).cpu().numpy().T.astype(np.int16)) 
                
    return 

def test_model(args, model, report_dir):

    model = model.eval()

    starting_point = int(args.test.starting_point)
    amount = int(args.test.amount)
    audio_pre_process = AudioPreProcessing(args)

    with torch.no_grad():

        for i in range(starting_point, starting_point + amount):

            path = str(ROOT_PATH)
            file_name = "file_" +  str(i)
            target_file = path + '/' + args.data_set_path + "lables/" + file_name + ".csv"
            feature_file = path + '/' + args.data_set_path + "preprocessing3/" + file_name + ".csv"

            output = inference(args, audio_pre_process, model, target_file, feature_file, report_dir, file_name)

            # write output file
            # write(output_file_generated, args.sample_rate, (output*2**15).cpu().numpy().T.astype(np.int16)) 
        
    return
