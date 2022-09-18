import numpy as np
import matplotlib.pyplot as plt
import gcc
import soundfile as sf #TODO

def frame_preprocessing(signal, mic_locations, number_of_direction, samplerate, type = "spectrum"):
    """
    preprocessing of the sounds recording before the neural network

    this preprocessing pass to the neural network the cross correlation 
        of the signals in certain directions.
    we summarize the cross correlations of all microphons 
        (depending on their locations of caurse)
    each feature is: 1 * number_of_direction

    Parameters
    ----------
    signal: numpy.ndarray 
        audio data
        shape: frame_size*num_of_mic
    mic_locations: numpy.ndarray
        (x,y,z) locations of the microphons in space
        shape: num_of_mic * 3
    number_of_direction: int
        The number of angles in space we would like to classify into
    samplerate: int
        sample rate
    type: str {'time', 'correlation', 'spectrum'}
        A string indicating the type of preprocessing 

        ``time``
            this preprocessing pass to the neural network the samples in time
            in fact there is no processing in this option
            each feature is: frame_size * number_of_mic
        ``correlation``
            this preprocessing pass to the neural network the cross correlation 
                of the signals in certain directions.
            we summarize the cross correlations of all microphons 
                (depending on their locations of caurse)
            each feature is: 1 * number_of_direction
        ``spectrum``
            this preprocessing pass to the neural network the spectrum 
                of the signals in certain directions.
            num_of_frequency will be the (frame_size/2 + 1),
                because of the symmetry in the frequency (the signals is real)
            we summarize the cross spectrum of all microphons 
                (depending on their locations of caurse)
            each feature is: (frame_size/2 + 1) * number_of_direction
    
    Returns
    -------    
    cross_correlation_array: numpy.ndarray 
        the cross correlation of the signals in certain directions
        shape: depends of the preprocessing type:
        ``time``:  frame_size * number_of_mic
        ``correlation``: 1 * number_of_direction
        ``spectrum``: num_of_frequency * number_of_direction

    """

    # check the function arguments
    if type not in ("time", "correlation", "spectrum"): 

        raise ValueError("Type must be one of: time, correlation, spectrum!")

    # preprocessing 1 return the signals in time
    if type == "time":
        return signal

    # the function will return cross_correlation_array
    elif type == "correlation": 
        return_val = np.zeros((1, number_of_direction), dtype = 'float64') 

    # the function will return spectral_image
    elif type == "spectrum":
        dimension_of_spectral_image = int(len(signal)/2 + 1)
        return_val = np.zeros((dimension_of_spectral_image, number_of_direction), dtype = 'float64') 

    num_of_mic = signal.shape[1]
    graund_truth_directions = np.linspace(0, 360, number_of_direction, endpoint=False)

    # loop on the of the microphons array
    for mic2_indx in range(1, num_of_mic):

        # mic1_indx = mic2_indx - 1 # choose the neighborhood index
        mic1_indx = 0

        # calculate the distance between the 2 microphons
        dist_bet_mic = gcc.dist(mic_locations[mic1_indx], mic_locations[mic2_indx])

        # rotate the directions to the relative angle of the 2 microphons
        angle_of_mic = gcc.angle(mic_locations[mic1_indx], mic_locations[mic2_indx])
        relaticve_direction = (graund_truth_directions - angle_of_mic) % 360

        # calculate the number of shift samples for each direction of the M directions
        times_list_array = gcc.angle_to_shift_samples(relaticve_direction, dist_bet_mic, samplerate)

        # preprocessing 2: calculate cross correaltion in M different times
        if type == "correlation":
            curr_val = gcc.cross_corr_by_freq(signal[:,mic1_indx], signal[:,mic2_indx], times_list_array)

        # preprocessing 3: calculate spectral image in M different times
        if type == "spectrum":
            curr_val = gcc.spectral_image(signal[:,mic1_indx], signal[:,mic2_indx], times_list_array)
                # #debug TODO
                # sum_val = curr_val.sum(axis=0)/1024

        # # debug correlation TODO 
        # print("curr angle is: {}".format(graund_truth_directions[np.argmax(curr_val.real)]))
        # plt.figure(1)
        # plt.plot(relaticve_direction, abs(curr_val)) #corelation
        # plt.title("abs cross_correlation")
        # plt.figure(2)
        # plt.plot(relaticve_direction, curr_val.real) #corelation
        # plt.title("real cross_correlation")
        # plt.figure(3)
        # plt.plot(relaticve_direction, curr_val.imag) #corelation
        # plt.title("imag cross_correlation")
        # plt.show()

        # # debug spectrum TODO 
        # plt.figure(1)
        # plt.plot(relaticve_direction, curr_val.sum(axis=0).real) #spectrum
        # plt.title("real cross_correlation")
        # print("curr angle is: {}".format(graund_truth_directions[np.argmax(curr_val.sum(axis=0).real)]))
        # plt.figure(2)
        # p1 = sns.heatmap(curr_val.real, vmax= 0.00002) #spectrum  #vmax= 0.00002
        # plt.show()

        # add the cross correaltion of the 2 microphones to the overall array 
        return_val += curr_val.real

    return return_val

def signal_preprocessing(signal, mic_locations, number_of_direction, samplerate, frame_size = 1024, overlap = 512, type = "spectrum"):
    """
    devide the signal to frames and do the preprocessing for each frame

    Returns
    -------    
    signal_preprocessing_matrix: numpy.ndarray 
        the preprcessing of the signals
        shape: num_of_frames * shape_of_preprocessing
            ``time``: (num_of_frames * frame_size * number_of_mic)
            ``correlation``: (num_of_frames * 1 * number_of_direction)
            ``spectrum``: (num_of_frames * num_of_frequency * number_of_direction)
    """

    signal_preprocessing_matrix = []
    alpha = 0.95
    block_idx = 0

    # loop on the frames 
    for block in gcc.blocks(signal, frame_size, overlap):

        # remove the end of the signal if its too short
        if len(block) < frame_size:
            break
        
        temp_frame_preprocessing = frame_preprocessing(block, mic_locations, number_of_direction, samplerate,  type)
        # # option to do preprocess2 by sum on preprocess3
        # temp_frame_preprocessing = temp_frame_preprocessing.sum(axis=0)
        # temp_frame_preprocessing = temp_frame_preprocessing.reshape(1,36)

        if type == "spectrum" or type == "correlation":
            if block_idx==0:
                avrg_frame_preprocessing = temp_frame_preprocessing

            avrg_frame_preprocessing = gcc.moving_average(avrg_frame_preprocessing, temp_frame_preprocessing, alpha)
  
            signal_preprocessing_matrix.append(avrg_frame_preprocessing)

        else:
            signal_preprocessing_matrix.append(temp_frame_preprocessing)

        block_idx += 1

    signal_preprocessing_matrix = np.array(signal_preprocessing_matrix)

    return signal_preprocessing_matrix

def main():

    ########
    # test
    ########

    # # test 1: 3 mics, 4, 2 samples shift - suppose to be 30,60 degrees. and 60 degrees at all. 1024 samples - use frame_preprocessing func
    # white_noise_sig = gcc.create_shifted_white_signals(1024, [4,2])
    # mic_locations = np.array(([0.1,0.1,0],[0.1,0,0],[0,0.1,0]))
    # x = frame_preprocessing(white_noise_sig, mic_locations, 24, 16000, type = "correlation")
    
    # # test 2: 2 mics, 2.332 samples shift - suppose to be 60 degrees. long record - use signal_preprocessing
    # file_record1 = 'Final_Proj/old_input/clean_records/record1.wav'
    # file_record2 = 'Final_Proj/old_input/clean_records/record1_delayed.wav'
    # data_record1, samplerate_record1 = sf.read(file_record1)
    # data_record2, samplerate_record2 = sf.read(file_record2)
    # data_record = np.concatenate((data_record1.reshape((-1,1)),data_record2.reshape((-1,1))), axis = 1)
    # mic_locations = np.array(([0.1,0.1,0],[0.1,0,0]))
    # x = signal_preprocessing(data_record[14000:,:], mic_locations, 24, 16000, frame_size = 1024, overlap = 512, type = "correlation")

    pass

if __name__ == '__main__':

    main()