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
# import matlab.engine as mt

def input_file_name_change(data_folder, prefix_name):

    """
    change names of all the files in the directory to serial name
    important: first need to check in the directory, that there is no file with the same name right now!
    """

    files = os.listdir(data_folder)

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

def add_noise(signal, SNR_ratio ,mics_num, mic_locations, VAD_array,  Noise = 'WhiteNoise'):
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

    #     #create INF filter from matlab code
        
    #     eng = mt.engine.start_matlab()
    #     INF_filter = eng.sinf_3D(mic_locations,len = 4096)
    #     eng.quit()

    #     array_signal_with_noise_3D =  ss.convolve(INF_filter[:, None, :], signal[:, :, None])
    #     array_signal_with_noise = array_signal_with_noise_3D.reshape(array_signal_with_noise_3D.shape[0],mics_num) #convert 3D array to 2D
       
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

    # const array parameters and RIR
    mics_num_const = 6
    mics_radius_const = 6/100 #cm to m

    # Room dimensions [x y z] (m)
    x_room, y_room, z_room = 6, 6, 2.4

    # mic array location
    r_const = [] #const receiver position(s) [x y z] (m)
    teta = 0 #mic location initialize

    # remove and create the directories
    remove_dirs(
        [data_folder + 'mics', data_folder + 'mics/random_array', data_folder + 'mics/fix_array',
        data_folder + 'RIR', data_folder + 'RIR/random_array', data_folder + 'RIR/fix_array',
        data_folder + 'meta', data_folder + 'meta/random_array', data_folder + 'meta/fix_array',
        data_folder + 'mics_position', data_folder + 'mics_position/random_array', data_folder + 'mics_position/fix_array',
        data_folder + 'VAD_lables', data_folder + 'VAD_lables/random_array', data_folder + 'VAD_lables/fix_array']
    )

    create_dirs(
        [data_folder + 'mics', data_folder + 'mics/random_array', data_folder + 'mics/fix_array',
        data_folder + 'RIR', data_folder + 'RIR/random_array', data_folder + 'RIR/fix_array',
        data_folder + 'meta', data_folder + 'meta/random_array', data_folder + 'meta/fix_array',
        data_folder + 'mics_position', data_folder + 'mics_position/random_array', data_folder + 'mics_position/fix_array',
        data_folder + 'VAD_lables', data_folder + 'VAD_lables/random_array', data_folder + 'VAD_lables/fix_array']
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

        print("Create file: " + str(i))

        # rand parameters
        angle = np.random.randint(1,361)
        mics_num = np.random.randint(4,13)
        mics_radius = np.random.randint(3,9)/100 #cm to m
        source_dist = np.random.randint(1,3)
        file_index = np.random.randint(1,16)
        rev_time  = np.random.randint(2,9)/10 # Reverberation time (s)  
        SNR_ratio = np.random.normal(18,3) #noraml distribution with mean of 18 and std of 3
        # SNR range is 0 to 40 dB
        if SNR_ratio < 0 : SNR_ratio = 0
        if SNR_ratio > 40 : SNR_ratio = 40

        # choose signal 
        signal_name = data_folder + 'oracle/file_' + str(file_index) + '.flac'
        pre_amp_signal, fs = sf.read(signal_name,always_2d=True)
        Frame_num = 190
        Frame_size = 1024

        while(pre_amp_signal.shape[0] < Frame_num*Frame_size):
            file_index = np.random.randint(1,16)
            signal_name = data_folder + 'oracle/file_' + str(file_index) + '.flac'
            pre_amp_signal, fs = sf.read(signal_name,always_2d=True)
        
        pre_amp_signal = pre_amp_signal[:Frame_num*Frame_size]

        # Amplify the signal to max of 0.9
        signal = pre_amp_signal * 0.9 / max(pre_amp_signal)

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

        # random array RIR generation
        x_source=source_dist*np.cos(angle*np.pi/180)+x_room/2
        y_source=source_dist*np.sin(angle*np.pi/180)+y_room/2

        h = rir.generate(
            c=340,                  # Sound velocity (m/s)
            fs=fs,                  # Sample frequency (samples/s)
            r=r,                     # Receiver position(s) [x y z] (m)  
            s=[x_source, y_source, 1.5],          # Source position [x y z] (m)
            L=[x_room, y_room, z_room],           # Room dimensions [x y z] (m)
            reverberation_time=rev_time,          # Reverberation time (s)
            nsample=4096,                         # Number of output samples
        )

        # fix array RIR generation
        h_const = rir.generate(
            c=340,                  # Sound velocity (m/s)
            fs=fs,                  # Sample frequency (samples/s)
            r=r_const,              # Receiver position(s) [x y z] (m)
            s=[x_source, y_source, 1.5],           # Source position [x y z] (m)
            L=[x_room, y_room, z_room],            # Room dimensions [x y z] (m)
            reverberation_time=rev_time,           # Reverberation time (s)
            nsample=4096,                          # Number of output samples
        )

        # Convolve 2-channel signal with *mix_num* impulse responses
        random_array_signal_3D = ss.convolve(h[:, None, :], signal[:, :, None])
        const_array_signal_3D = ss.convolve(h_const[:, None, :], signal[:, :, None])
        random_array_signal_2D = random_array_signal_3D.reshape(random_array_signal_3D.shape[0],mics_num) #convert 3D array to 2D
        const_array_signal_2D = const_array_signal_3D.reshape(const_array_signal_3D.shape[0],mics_num_const) #convert 3D array to 2D

        # calculate the voice activity
        VAD_array = voice_activity_detection(random_array_signal_2D, win_size = 1024, win_hop=512, threshold=-50, E0=max(random_array_signal_2D[:,0]))

        # add noise
        random_array_signal_with_noise = add_noise(random_array_signal_2D, SNR_ratio, mics_num, r, VAD_array,  Noise = 'WhiteNoise')
        const_array_signal_with_noise = add_noise(const_array_signal_2D, SNR_ratio, mics_num_const, r_const, VAD_array, Noise = 'WhiteNoise')

        # save signal.wav files to mics folder
        sf.write(data_folder + 'mics/random_array/file_'+str(i)+'.wav',random_array_signal_with_noise,fs) 
        sf.write(data_folder + 'mics/fix_array/file_'+str(i)+'.wav',const_array_signal_with_noise,fs)

        # save RIR.wav filter files to RIR folder
        sf.write(data_folder + 'RIR/random_array/file_'+str(i)+'.wav',r,fs) 
        sf.write(data_folder + 'RIR/fix_array/file_'+str(i)+'.wav',r_const,fs)

        # save labels output files to meta folder
        header_rand = ['file_index','angle','mics_num','mics_radius','source_dist','rev_time','SNR_ratio']
        data_rand = [file_index,angle,mics_num,mics_radius,source_dist,rev_time,SNR_ratio]
        data_path_rand = data_folder + 'meta/random_array/file_meta.csv'

        header_fix = ['file_index','angle','mics_num_const','mics_radius_const','source_dist','rev_time','SNR_ratio']
        data_fix = [file_index,angle,mics_num_const,mics_radius_const,source_dist,rev_time,SNR_ratio]
        data_path_fix = data_folder + 'meta/fix_array/file_meta.csv'

        write_meta(header_fix,data_fix,data_path_fix,i)
        write_meta(header_rand, data_rand,data_path_rand,i)

        # save mics position output files to mics_position folder
        header_rand = ['x','y','z']
        data_rand = np.resize(r,(mics_num,3))
        data_path_rand = data_folder + 'mics_position/random_array/file_'+str(i)+'.csv'

        header_fix = header_rand
        data_fix = np.resize(r_const,(mics_num_const,3))
        data_path_fix = data_folder + 'mics_position/fix_array/file_'+str(i)+'.csv'

        write_mics_position(header_fix,data_fix,data_path_fix)
        write_mics_position(header_rand, data_rand,data_path_rand)

        # save VAD lables output files to VAD_lables folder
        lables = list([angle,1*x] for x in VAD_array)
        data_path_fix = data_folder + 'VAD_lables/fix_array/file_'+str(i)+'.csv'
        data_path_rand = data_folder + 'VAD_lables/random_array/file_'+str(i)+'.csv'
        write_VAD(lables,data_path_fix)
        write_VAD(lables,data_path_rand)

def main():

    # change input file name to 'file_1.flac' - only at the first time
    change_input_file_name = False
    if change_input_file_name:
        input_file_name_change('data/oracle/', 'file_')

    # create the signal with rir generator
    signal_gen("data/", 15)

if __name__ == '__main__':

    main()