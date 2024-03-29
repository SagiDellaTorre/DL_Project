import numpy as np
import os
import torch
from scipy.io.wavfile import write
import sys
import soundfile as sf
import matplotlib.pyplot as plt
from data.data_utils import AudioPreProcessing
from pathlib import Path
import csv
ROOT_PATH = Path(__file__).parent.parent.parent

def create_dirs(dirs_list):
    """
    create the directories, if they aren't exist
    """
    for directory in dirs_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def inference(args, audio_pre_process, model, target_wav, feature_wav, report_dir, file_name, signal_wav):

    # the real signal
    signal, samplerate = sf.read(signal_wav)
    frame_jump = args.window_size - args.overlap

    # plot and calculate the error only after the network is converging
    start_from_frame = int(args.test.start_from_frame)
    starting_points_to_plot_signal = start_from_frame * frame_jump

    # calculate the output of the network
    target, input_data = audio_pre_process.transformations(target_wav,feature_wav)

    if torch.cuda.is_available():
        input_data = input_data.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))

    if args.pre_process_type == 'analytic':
        pred = torch.argmax(abs(input_data), dim = 1).reshape(1,-1,1)/args.number_of_direction
    else:
        pred = model(torch.unsqueeze(input_data, 0))

    ## calculate the error
    abs_error_tensor = abs(pred[0,start_from_frame:,0]*360 - target[start_from_frame:,0])
    mean_absolute_error = sum(torch.minimum(abs_error_tensor, 360 - abs_error_tensor))/(pred.shape[1] - args.test.start_from_frame)
    mean_absolute_error = round(float(mean_absolute_error), 2)

    ## plot the results
    target_on_cpu = target.cpu()
    pred_on_cpu = pred.cpu()

    # Visualize spectogram - figure without VAD
    fig, axs = plt.subplots(2, 1, constrained_layout=True)   
    fig.suptitle('NN predictions. Mean Absolute Error = ' + str(mean_absolute_error)) 

    axs[0].plot(range(signal.shape[0] - starting_points_to_plot_signal), signal[starting_points_to_plot_signal:,1]*0.9/max(signal[starting_points_to_plot_signal:,1]), label="signal")
    axs[0].set_title('VAD figure')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('VAD')
    axs[0].legend()

    axs[1].plot(range(target_on_cpu.shape[0]-start_from_frame), target_on_cpu[start_from_frame:,0], label="target")
    axs[1].plot(range(target_on_cpu.shape[0]-start_from_frame), pred_on_cpu[0,start_from_frame:,0]*360, label="prediction")
    # axs[1].plot(range(target_on_cpu.shape[0]), target_on_cpu[:,1]*360, label="VAD")
    axs[1].set_title('Angle figure')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Angle')
    axs[1].legend()

    plt.savefig(report_dir + file_name + '.png')

    if args.pre_process_type != 'analytic':
        #add the VAD to the figure
        axs[0].plot(range(target_on_cpu.repeat_interleave(frame_jump, dim=0).shape[0]- starting_points_to_plot_signal), target_on_cpu.repeat_interleave(frame_jump, dim=0)[starting_points_to_plot_signal:,1], label="target")
        axs[0].plot(range(pred_on_cpu.repeat_interleave(frame_jump, dim=1).shape[1] - starting_points_to_plot_signal), pred_on_cpu.repeat_interleave(frame_jump, dim=1)[0,starting_points_to_plot_signal:,1], label="prediction")

    plt.savefig(report_dir + file_name + 'with_VAD'+ '.png')

    return mean_absolute_error


def test_model(args, model, report_dir):

    if model:
        model = model.eval()

    amount = int(args.test.amount)
    audio_pre_process = AudioPreProcessing(args)

    path = str(ROOT_PATH)
    pre_process_type = 'preprocessing2' if args.pre_process_type == 'analytic' else args.pre_process_type
    rec_type = "random_array" if args.test_type == 'rir' else "fix_array"
    test_feature_dir = path + '/' + args.test_set_path  + pre_process_type + "/"
    files = os.listdir(test_feature_dir)

    error_list = {}

    with torch.no_grad():

        # for all tiles in the test dir:
        for i, file in enumerate(files):

            if amount != -1 and i >= amount:
                break

            file_name = file.rsplit('.',1)[0]
            target_file = path + '/' + args.test_set_path + "lables/" + file_name + ".csv"
            feature_file = path + '/' + args.test_set_path + pre_process_type + "/" + file_name + ".csv"
            signal = path + '/' + args.test_set_path + "../mics/" + rec_type + "/" + file_name + ".wav"

            # use the network, plot the results and calculate the error
            output = inference(args, audio_pre_process, model, target_file, feature_file, report_dir, file_name, signal)

            error_list[file_name] = output

        # calculate the average error
        res = 0
        for val in error_list.values():
            res += val
        average_for_all_files = res / len(error_list)
        error_list["Average"] = average_for_all_files

        # write results to csv
        with open(report_dir + 'error_results' + '.csv', 'w') as output:
            writer = csv.writer(output)
            writer.writerow(["File Name", "Mean Absolute Error"])
            for key, value in error_list.items():
                writer.writerow([key, value])

    return average_for_all_files
