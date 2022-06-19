import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt
import csv as csv
import os
import sys
import shutil
sys.path.append('src/features')
import gcc
import pandas as pd
import time
# import matlab.engine as mt

def input_file_name_change(data_folder, prefix_name):

    """
    change names of all the files in the directory to serial name
    important: first need to check in the directory, that there is no file with the same name right now!
    """

    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    for index, file in enumerate(files):
        os.rename(os.path.join(data_folder, file), os.path.join(data_folder, ''.join([prefix_name, str(index + 1), '.flac'])))

def create_dirs(dirs_list):
    """
    create the directories, if they aren't exist
    """
    for directory in dirs_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def remove_dirs(dirs_list):
    """
    remove the directories, if they are exist
    """
    for directory in dirs_list:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def write_mics_position(header,data,data_path):

    # open the file in the write mode
    with open(data_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # write the header
        writer.writerow(header)

        # write the data
        for mics_pos in data:
            writer.writerow(mics_pos)
    # close the file
    file.close()

def write_VAD(VAD_array,data_path):

    df = pd.DataFrame(VAD_array)
    df.to_csv(data_path, index=True)

def write_meta(header,data,data_path,i):

    # open the file in the write mode
    with open(data_path, 'a', newline='') as file:
        writer = csv.writer(file)

        # write the header
        if i == 0:
            writer.writerow(header)

        # write the data
        writer.writerow(data)
    # close the file
    file.close()

def SNR(signal,noise,VAD_array):
    """
    calculate SNR
    """ 
    signal_energy = (np.std(signal[VAD_array], axis = 0))**2
    noise_energy = (np.std(noise[VAD_array], axis = 0))**2
    SNR = np.mean(10*np.log10(signal_energy/noise_energy))

    return SNR

def add_noise(signal, noise, SNR_ratio ,mics_num, VAD_array,  Noise = 'RoomNoise'):
    """
    add noise to the signals
    """    

    if Noise == 'WhiteNoise': ## Add white noise by SNR

        # array_std = np.std(signal, axis = 0)
        array_std = np.std(signal[VAD_array], axis = 0)

        # normal noise std
        alpha = array_std / np.sqrt(10**(SNR_ratio/10))
        alpha = np.resize(alpha, (signal.shape))

        # matrix for normalize whiteNoise
        array_WhiteNoise = np.random.normal(0, 1,(signal.shape[0], mics_num))
        array_WhiteNoise_normalized = np.multiply(alpha, array_WhiteNoise)

        # add noise to signal
        array_signal_with_noise =  signal + array_WhiteNoise_normalized 

        # # sanity check
        # signal_energy = (np.std(signal[VAD_array], axis = 0))**2
        # noise_energy = (np.std(array_WhiteNoise_normalized, axis = 0))**2
        # SNR = 10*np.log10(signal_energy/noise_energy)

    # if Noise == 'INF': #isotropic noise field

    if Noise == 'RoomNoise': ## Add Room noise by SNR = 10log(signal/(alpha*noise))

        # array_std = np.std(signal, axis = 0)
        array_std = np.std(signal[VAD_array], axis = 0)/np.std(noise[VAD_array], axis = 0) #signal/noise

        # room noise std
        alpha = array_std / np.sqrt(10**(SNR_ratio/10))
        alpha = np.resize(alpha, (signal.shape))

        # matrix for normalize RoomNoise
        array_RoomNoise_normalized = np.multiply(alpha, noise)

        # add noise to signal
        array_signal_with_noise =  signal + array_RoomNoise_normalized 
        
        # # sanity check
        # signal_energy = (np.std(signal[VAD_array], axis = 0))**2
        # noise_energy = (np.std(array_RoomNoise_normalized, axis = 0))**2
        # SNR = 10*np.log10(signal_energy/noise_energy)
   
    return array_signal_with_noise

def voice_activity_detection(signal, win_size = 1024, win_hop=512, threshold=-30, E0=1):
    """
    Distinction of voiced segments from silent ones.
    We frame the signal, compute the short time energy aka the energy pro frame and according to a predefined threshold we can decide where the frame is voiced or silent

    E0: Energy value characterizing the silence to speech energy ratio

    Returns
    -------    
    VAD_array: 
        array in the length of the signal.
        0 - for noise, and 1 for speech.
    """   

    enrgy_array = np.zeros((len(signal),2))
    idx = 0

    for block in gcc.blocks(signal, win_size, win_hop):

        block_flat = block[:,0]
        norm_energy = np.sum(np.abs(np.fft.rfft(a=block_flat, n=win_size))**2, axis=-1) / win_size**2
        log_energy = 10 * np.log10(norm_energy / E0)

        enrgy_array[idx:idx+len(block_flat),0] += log_energy
        enrgy_array[idx:idx+len(block_flat),1] += 1

        idx = idx + win_hop

    avrg_enrgy_array = enrgy_array[:,0]/enrgy_array[:,1]

    VAD_array = np.array(avrg_enrgy_array > threshold)

    # plt.figure(1)
    # plt.plot(signal)
    # plt.plot(VAD_array)
    # plt.title("VAD - Voice Activity Detection")
    # plt.show()

    return VAD_array

def signal_gen(data_folder, signals_num):

    number_of_oracle_files = signals_num
    RECdata = pd.read_csv(data_folder + 'oracle/RECmeas0.csv')

    # const array parameters and RIR
    mics_num_const = 6
    mics_radius_const = 5/100 #cm to m

    # Room dimensions [x y z] (m)
    x_room, y_room, z_room = 6, 10, 2.4

    # mic array location
    r_const = [] #const receiver position(s) [x y z] (m)
    teta = 0 #mic location initialize

    # remove and create the directories
    remove_dirs(
        [data_folder + 'RIR', data_folder + 'RIR/fix_array',
        data_folder + 'meta', data_folder + 'meta/fix_array',
        data_folder + 'mics_position', data_folder + 'mics_position/fix_array',
        data_folder + 'VAD_lables', data_folder + 'VAD_lables/fix_array']
    )

    create_dirs(
        [data_folder + 'RIR', data_folder + 'RIR/fix_array',
        data_folder + 'meta', data_folder + 'meta/fix_array',
        data_folder + 'mics_position', data_folder + 'mics_position/fix_array',
        data_folder + 'VAD_lables', data_folder + 'VAD_lables/fix_array']
    )
   
    #const receiver position(s) [x y z] (m) initialize
    for j in range(mics_num_const):
        mic_location_const = [] #Declaring an empty 1D array.
        mic_location_const.append(mics_radius_const*np.cos(teta*np.pi/180)+x_room/2) #append x axis room mic location
        mic_location_const.append(mics_radius_const*np.sin(teta*np.pi/180)+y_room/2) #append y axis room mic location
        mic_location_const.append(1.5)                                     #append z axis room mic location
        teta = teta + 360/mics_num_const   #teta update for next mic
        r_const.append(mic_location_const) #append [x,y,z] specific mic to mics array

    for i in range (signals_num):

        if i % 100 == 0:
            print("Create file: " + str(i))

        # const parameters
        angle = RECdata.angle[i]
        mics_num = RECdata.mics_num[i]   
        mics_radius = RECdata.mics_radius[i]/100 #cm to m
        source_dist = RECdata.source_dist[i]
        rev_time  = RECdata.rev_time[i] # Reverberation time (s)  

        # SNR_ratio = np.random.normal(18,3) #noraml distribution with mean of 18 and std of 3
        # # SNR range is 0 to 40 dB
        # if SNR_ratio < 0 : SNR_ratio = 0
        # if SNR_ratio > 40 : SNR_ratio = 40

        # choose signal 
        signal_name = data_folder + 'oracle/file_' + str(i) + '.flac'
        pre_amp_signal, fs = sf.read(signal_name,always_2d=True)
        frame_num = 190
        frame_size = 1024

        # if the signal is too short - fill it (cyclic)
        while(pre_amp_signal.shape[0] < frame_num*frame_size):
            pre_amp_signal = np.concatenate((pre_amp_signal, pre_amp_signal), axis = 0)
        
        # if the signal is too long - cut it
        pre_amp_signal = pre_amp_signal[:frame_num*frame_size]

        # # Amplify the signal to max of 0.9
        # signal = pre_amp_signal.all() * 0.9 /max(pre_amp_signal.all())
        signal = pre_amp_signal

        # mic array location
        r = [] #Receiver position(s) [x y z] (m)
        teta = 0 #mic location initialize

        # const receiver position(s) [x y z] (m) initialize
        for j in range(mics_num):
            mic_location = [] #Declaring an empty 1D array.
            mic_location.append(mics_radius*np.cos(teta*np.pi/180)+x_room/2) #append x axis room mic location
            mic_location.append(mics_radius*np.sin(teta*np.pi/180)+y_room/2) #append y axis room mic location
            mic_location.append(1.5)                               #append z axis room mic location
            teta = teta + 360/(mics_num) #teta update for next mic
            r.append(mic_location)     #append [x,y,z] specific mic to mics array

        # calculate the voice activity
        VAD_array = voice_activity_detection(signal, win_size = 1024, win_hop=512, threshold=-50, E0=max(signal[:,0]))

        #choose noise record
        noise_name = data_folder + 'oracle/quiet_room.flac'
        #noise_name = data_folder + 'oracle/aircondition_noise_room.flac'
        noise_record, fs = sf.read(noise_name,always_2d=True)

        # if the noise is too short - fill it (cyclic)
        while(noise_record.shape[0] < frame_num*frame_size):
            pre_amp_signal = np.concatenate((noise_record, noise_record), axis = 0)
        
        # if the noise is too long - cut it
        noise_record = noise_record[:frame_num*frame_size]

        # calculte SNR 
        SNR_ratio = SNR(signal,noise_record, VAD_array)

        # add noise
        SNR_art = [20,15,10,6,3,0] #SNR artificial 
        for j in range (len(SNR_art)):
            const_array_signal_with_noise = add_noise(signal, noise_record, SNR_art[j], mics_num_const, VAD_array, Noise = 'RoomNoise')
            sf.write(data_folder + 'mics/fix_array/file_'+str(i)+'_'+str(SNR_art[j])+'dB_SNR.wav',const_array_signal_with_noise,fs)

        # save signal.wav files to mics folder
        sf.write(data_folder + 'mics/fix_array/file_'+str(i)+'.wav',signal,fs)

        # save RIR.wav filter files to RIR folder
        sf.write(data_folder + 'RIR/fix_array/file_'+str(i)+'.wav',r_const,fs)

        # save labels output files to meta folder
        header_fix = ['file_index','angle','mics_num_const','mics_radius_const','source_dist','rev_time','SNR_ratio']
        data_fix = [i,angle,mics_num_const,mics_radius_const,source_dist,rev_time,SNR_ratio]
        data_path_fix = data_folder + 'meta/fix_array/file_meta.csv'

        write_meta(header_fix,data_fix,data_path_fix,i)

        # save mics position output files to mics_position folder
        header_fix = ['x','y','z']
        data_fix = np.resize(r_const,(mics_num_const,3))
        data_path_fix = data_folder + 'mics_position/fix_array/file_'+str(i)+'.csv'
        write_mics_position(header_fix,data_fix,data_path_fix)
        for j in range (len(SNR_art)):
            data_path_fix = data_folder + 'mics_position/fix_array/file_'+str(i)+'_'+str(SNR_art[j])+'dB_SNR.csv'
            write_mics_position(header_fix,data_fix,data_path_fix)

        # save VAD lables output files to VAD_lables folder
        lables = list([angle,1*x] for x in VAD_array)
        data_path_fix = data_folder + 'VAD_lables/fix_array/file_'+str(i)+'.csv'
        write_VAD(lables,data_path_fix)
        for j in range (len(SNR_art)):
            data_path_fix = data_folder + 'VAD_lables/fix_array/file_'+str(i)+'_'+str(SNR_art[j])+'dB_SNR.csv'
            write_VAD(lables,data_path_fix)

def main():

    start = time.time()

    number_of_files_to_create = 4
    # create the signal with rir generator
    signal_gen("data/RECtest/", number_of_files_to_create)

    # print run time
    end = time.time()
    time_sec = round(end - start)
    time_min = round(time_sec / 60, 2)
    print("REC generator for " + str(number_of_files_to_create) + " files, took " + str(time_min) + " minutes.")

if __name__ == '__main__':

    main()