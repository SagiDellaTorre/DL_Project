import math
import numpy as np
from numpy.core.fromnumeric import argmax
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import interpolate, signal, fftpack

SPEED_OF_SOUND = 343 #meters per second

def blocks(signal, blocksize, overlap=0):
    """
    Return a generator for block-wise analize - each time block of signal.

    Parameters
    ----------
    signal: numpy.ndarray 
        long signal, which we want to dvide to blocks
        shape: length_of_signal*num_of_mic
    blocksize : int
        The number of frames to read per block.
    overlap : int, optional
        The number of frames to rewind between each block.
    """
    length = len(signal)
    i=0
    while i < length:
        yield signal[i:min(i+blocksize,length),:]
        i = i + blocksize - overlap 

def create_shifted_white_signals(length, delays_array):
    """
    return shifted white noises 
    they are equals except for the delay between them 
    Parameters
    ----------
    length: the length of the white noise
    delays_array: array
        array of the shifted signal delays ralated to the first signal

    white_noise_sig: ndarray
        the first column is the white noise
        the other columns is the shifted signals related to the first column
        shape: (length, len(delays_array) + 1)
    """

    mean = 0
    std = 1 
    num_samples = length
    num_of_signals = len(delays_array) + 1
    
    white_noise_sig = np.zeros((num_samples, num_of_signals))
    white_noise_sig[:,0] = np.random.normal(mean, std, size=num_samples)

    for i in range(num_of_signals - 1):
        shifted_white_noise_sig = np.roll(white_noise_sig[:,0], delays_array[i])
        white_noise_sig[:,i+1] = shifted_white_noise_sig

    return white_noise_sig

def delay_to_signal(signal, delay):
    
    if delay>=0:
        delayed_signal = np.concatenate((np.zeros(delay),signal[:len(signal)-delay]))
    else:
        delayed_signal = np.concatenate((signal[abs(delay):len(signal)],np.zeros(abs(delay))))

    return delayed_signal

def dist(a, b):

    return math.sqrt( (b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)

def angle(point_a, point_b):
    """
    return degree between vector of two points to the x-axis
        degree between 0 to 360.
        the rotate is counterclockwise.
    """

    x_dist = (point_a[0] - point_b[0])
    y_dist = (point_a[1] - point_b[1])

    angle_rad = math.atan2(y_dist, x_dist)
    angle_deg = angle_rad*180/math.pi
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def moving_average(cc_old, cc_new, alpha=0.95):
    """
    Average of new cross coralation calculated with the old one

    alpha is the weight to give to the old cross correlation 

    alpha ^ n is the decay בoefficient  
    """
    return  alpha * cc_old + (1 - alpha) * cc_new

def ifft_changeable_points(x, times_list):

    K = len(x)
    y = np.zeros(len(times_list),dtype=complex)

    for i, n in enumerate(times_list):
        y[i] = np.sum(x * np.exp(2j * np.pi * n * np.arange(K)/K)) / K

    return y

def cross_corr_by_freq(signal1, signal2, times_list = None):
    """
    calculate the cross correlatins in fix points
    method:    
        move to frequency, 
        calculate the cross spectrum by multiply the fft's,
        do ifft in specific points (times list)

    Parameters
    ----------
    signal1,signal2: two array data
    times_list: array of points in which to calculate the cross correlation

    Returns
    -------    
    cross_corr: cross correlation in the specific points
    """

    signal1_k = fftpack.fft(signal1)
    signal2_k = fftpack.fft(signal2)
    cross_spectrum = signal2_k * np.conj(signal1_k)
    cross_spectrum_shifted_to_center = fftpack.fftshift(cross_spectrum)
    
    # #debug cc
    # plt.figure(4)
    # plt.plot(signal1_k) #corelation
    # plt.figure(5)
    # plt.plot(signal2_k) #corelation
    # plt.figure(6)
    # plt.plot(cross_spectrum) #corelation
    # plt.figure(7)
    # plt.plot(cross_spectrum_shifted_to_center) #corelation
    # plt.show()

    if times_list is None:
        times_list = np.arange(start=-4,stop=5)

    cross_corr = ifft_changeable_points(cross_spectrum_shifted_to_center, times_list)

    # #debug cc
    # cross_corr2 = fftpack.ifft(cross_spectrum)
    # real_cross_correlation = signal.correlate(signal1,signal2)
    # angles_list = np.arange(start=0,stop=360,step=15) #np.arange(start=0,stop=190,step=1)
    # fig, axs = plt.subplots(3, 1, constrained_layout=True)
    # axs[0].plot(angles_list, cross_corr, 'o')
    # axs[0].set_title('cross correlation calculated from IDFT from the cross spectrum')
    # axs[0].set_xlabel('angles (not an integer number of samples)')
    # axs[0].set_ylabel('cross correlation')
    # axs[1].plot(real_cross_correlation, 'o') #[1019:1028]
    # axs[1].set_title('cross correlation calculated directly from defenition - only 4 samples from each size of the center')
    # axs[1].set_xlabel('num of samples')
    # axs[1].set_ylabel('cross correlation')
    # axs[2].plot(cross_corr2, 'o')
    # axs[2].set_title('cross correlation calculated from cross spectrum - only 4 samples from each size of the center')
    # axs[2].set_xlabel('num of samples')
    # axs[2].set_ylabel('cross correlation')
    # fig.show()

    return cross_corr

def spectral_image(signal1, signal2, times_list = None):
    """
    calculate the spectral image in fix points
    method:    
        move to frequency, 
        calculate the cross spectrum by multiply the fft's,
        multiply with exponents matrix in specific points (times list)
        number of frequency is (len(signal) / 2 + 1)
            because of the symmetry in the frequency (the signals is real)

    Parameters
    ----------
    signal1,signal2: two array data
    times_list: array of points in which to calculate the cross correlation

    Returns
    -------    
    spectral_image: ndarray
        spectral image in the specific points
        shape: (len(signal)/2 + 1) * len(times_list)
    """

    signal1_k = fftpack.fft(signal1)
    signal2_k = fftpack.fft(signal2)
    cross_spectrum = signal2_k * np.conj(signal1_k)
    cross_spectrum_cut = cross_spectrum[:int(len(cross_spectrum)/2 + 1)]
    
    if times_list is None:
        times_list = np.arange(start=-4,stop=5)

    K = len(cross_spectrum) # frequency
    positive_frequency = len(cross_spectrum_cut) # symetry in frequency
    spectral_image = np.zeros((positive_frequency, len(times_list)),dtype=complex)

    for i, n in enumerate(times_list):
        spectral_image[:,i] = cross_spectrum_cut * np.exp(2j * np.pi * n * np.arange(positive_frequency)/K)

    return spectral_image

def partial_crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return value has length 2*maxlag + 1.
    """
    # TODO: inefficient (calculate all the crros correlation, but take only part of the samples)
    cross_corr = signal.correlate(x,y,mode = 'full')[len(x)-maxlag-1:len(x)+maxlag] 

    return cross_corr

def improve_cc_resolution_by_interpulation(cross_corr, increase_rate):
    """
    improve the resolution of the cross correlation to recognize more angles

    Parameters
    ----------
    cross_corr: the cross correlation in the old rate
    increase_rate: how many times to increase the rate of the cross correlation

    Returns
    -------    
    new_cross_corr: discrate array of the cross correlation in the new rete
        the length of the new cross correlation is: ((len(cross_corr)-1)*increase_rate) + 1
    """

    x = np.arange(0, len(cross_corr))
    new_cross_corr = interpolate.interp1d(x, cross_corr, kind = 'cubic')

    x_new = np.arange(0, len(cross_corr)-1,1/increase_rate)

    return new_cross_corr(x_new)

def cc_to_angle(cross_corr, dist_between_mic, samplerate):
    """
    Provide DOA from cross correlation of two signals

    Parameters
    ----------
    cross correlation: cross correlation of audio data    
    dist_between_mic: the distance between the two microphones
    samplerate: semplerate of the audio data

    Returns
    -------    
    angle: angle between the two signals in radians
    In case of error (there is no angle which solve the problem) the angle that returned is -10
    
    """

    max_cc = np.argmax(cross_corr) #peak of the cross correlation
    sample_shift = int(len(cross_corr)/2 - max_cc)
    time_delay = sample_shift / samplerate #seconds
    distance_delay = time_delay * SPEED_OF_SOUND

    if(abs(distance_delay)<dist_between_mic):
        angle = math.acos(distance_delay/dist_between_mic)
    else:
        angle = -2*math.pi # error value = -360 degrees
    
    return angle

def angle_to_shift_samples(angles_list, dist_between_mic, samplerate):
    """
    Get list of angles, and match to each angle num of shifted samples respectively

    Parameters
    ----------
    angles_list: list of angles in degrees
    dist_between_mic: the distance between the two microphones
    samplerate: samplerate of the audio data

    Returns
    -------    
    sample_shift_list: list of the shifted samples for each angle 
    
    """
    angles_list_rad = angles_list * math.pi / 180
    distance_delay = dist_between_mic * np.cos(angles_list_rad)
    time_delay = distance_delay / SPEED_OF_SOUND

    sample_shift_list = time_delay * samplerate

    return sample_shift_list

def doa_by_freq(signal1,signal2, dist_between_mic, samplerate):

    """
    Provide DOA from two signals for each block of the data
    Methos: move to frequency (calculate cross spectrum), 
    and than calculate ifft to came back to the fime domain (cross correlation)

    Parameters
    ----------
    signal1,signal2: two audio data
    dist_between_mic: the distance between the two microphones
    samplerate: samplerate of the aoudio data
    increase_sample_rate: int 
        how many time to increase the samplerate 
        default is 1 (leave it as is)
    """
    print()
    print("doa by frequency")

    #constants
    step_size = 1024
    overlap = 512
    max_num_of_corr_shift = 20
    alpha = 0.95

    # concatenate the 2 signals to one matrix of 2 dimensions
    signal = np.concatenate((signal1.reshape((-1,1)),signal2.reshape((-1,1))), axis = 1)

    angles_list = np.arange(start=0,stop=190,step=5)
    times_list = angle_to_shift_samples(angles_list, dist_between_mic, samplerate)

    #variables for the for loop
    avrg_cross_correlation = np.zeros(len(angles_list))
    block_idx = 0
    last_angle=-2 #initial value 

    for block in blocks(signal, step_size, overlap):

        if len(block) < (max_num_of_corr_shift*2 + 1): #cross correlation will be too short
            break

        temp_cross_correlation = cross_corr_by_freq(block[:,0], block[:,1], times_list)

        #moving average of the cross correlation
        avrg_cross_correlation = moving_average(avrg_cross_correlation, temp_cross_correlation, alpha) 
        if block_idx==0:
            avrg_cross_correlation = temp_cross_correlation

        angle = angles_list[argmax(abs(avrg_cross_correlation))]
        
        if (angle != last_angle):
            print()
            print("Block index:", block_idx, "(", block_idx*overlap/samplerate, "seconds). Angle:", angle , "degrees")
            last_angle = angle
 
        block_idx += 1

def doa_by_cross_corr(signal1,signal2, dist_between_mic, samplerate, increase_samplerate = 1):

    """
    Provide DOA from two signals for each block of the data

    Parameters
    ----------
    signal1,signal2: two audio data
    dist_between_mic: the distance between the two microphones
    samplerate: semplerate of the aoudio data
    increase_sample_rate: int 
        how many time to increase the samplerate 
        default is 1 (leave it as is)
    """
    print()
    print("doa by cross correlation")

    #constants
    step_size = 1024
    overlap = 512
    max_num_of_corr_shift = 20
    alpha = 0.95

    # concatenate the 2 signals to one matrix of 2 dimensions
    signal = np.concatenate((signal1.reshape((-1,1)),signal2.reshape((-1,1))), axis = 1)

    if type(increase_samplerate) is not int:
        raise ValueError('increase_samplerate must be an integer')

    #variables for the for loop
    avrg_cross_correlation = np.zeros(max_num_of_corr_shift * 2 * increase_samplerate)
    block_idx = 0
    last_angle=-2 #initial value 

    for block in blocks(signal, step_size, overlap):

        if len(block) < (max_num_of_corr_shift*2 + 1): #cross correlation will be too short
            break

        temp_cross_correlation = partial_crosscorrelation(block[:,0], block[:,1], max_num_of_corr_shift)
        temp_cross_correlation_high_rate = improve_cc_resolution_by_interpulation(temp_cross_correlation, increase_samplerate)

        #moving average of the cross correlation
        avrg_cross_correlation = moving_average(avrg_cross_correlation, temp_cross_correlation_high_rate, alpha) 
        if block_idx==0:
            avrg_cross_correlation = temp_cross_correlation_high_rate

        angle = cc_to_angle(avrg_cross_correlation, dist_between_mic, samplerate * increase_samplerate)

        if (angle != last_angle):
            print()
            print("Block index:", block_idx, "(", block_idx*overlap/samplerate, "seconds). Angle:", format(angle*180/math.pi, ".2f") , "degrees")
            last_angle = angle
 
        block_idx += 1

def main():

    ##############################
    #signal 1: white noise
    ##############################

    white_noise_sig = create_shifted_white_signals(1024, [4])
    # #suppose to be 30 degrees to 12 sample shift for 48kHz samplerate (or 4 for 16kHz)
    signal1 = white_noise_sig[:,0]
    signal2 = white_noise_sig[:,1]
    # doa_by_cross_corr(signal1, signal2, 0.1, 16000, increase_samplerate = 3)
    # doa_by_freq(signal1, signal2, 0.1, 16000)

    ##############################
    #signal 2: clean records
    ##############################

    file_record1 = 'Final_Proj/input/clean_records/record1.wav'

    data_record1, samplerate_record1 = sf.read(file_record1)
    data_record2 = delay_to_signal(data_record1, 4)
    # doa_by_cross_corr(data_record1, data_record2, 0.1, samplerate_record1)
    # doa_by_freq(data_record1, data_record2, 0.1, samplerate_record1)

    ##############################
    #signal 3: clean records - delayed sub samples
    ##############################

    file_record1 = 'Final_Proj/input/clean_records/record1.wav'
    file_record2 = 'Final_Proj/input/clean_records/record1_delayed.wav'

    data_record1, samplerate_record1 = sf.read(file_record1)
    data_record2, samplerate_record2 = sf.read(file_record2)
    # #2.332 samples shift - suppose to be 60 degrees
    # doa_by_cross_corr(data_record1, data_record2, 0.1, samplerate_record1, increase_samplerate=10)
    # doa_by_freq(data_record1, data_record2, 0.1, samplerate_record1)

    ##############################
    #signal 4: simulation records
    ##############################

    angles_list = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    angle_to_run = 165
    dist_between_mic_sim = 0.1

    file_sim_mic1 = 'Final_Proj/input/RIR_room_records/y1_' + str(angle_to_run) + '_16k.wav'
    file_sim_mic2 = 'Final_Proj/input/RIR_room_records/y2_' + str(angle_to_run) + '_16k.wav'

    data_sim_mic1, samplerate_sim_mic1 = sf.read(file_sim_mic1)
    data_sim_mic2, samplerate_sim_mic2 = sf.read(file_sim_mic2)

    # doa_by_cross_corr(data_sim_mic1, data_sim_mic2, dist_between_mic_sim, samplerate_sim_mic1, increase_samplerate = 3)
    # doa_by_freq(data_sim_mic1, data_sim_mic2, dist_between_mic_sim, samplerate_sim_mic1)

    ##############################
    #signal 5: our records
    ##############################

    MIC_LOCATION_ARRAY = [0, 0.022, 0.039, 0.077, 0.098, 0.116]
    file_to_read = 'Final_Proj/input/dvir_records/30.wav'
    mic_idx_to_check = [3,4]

    #read the file, and get the data and the samplerate
    data, samplerate = sf.read(file_to_read)

    signal1 = data[:,mic_idx_to_check[0]]
    signal2 = data[:,mic_idx_to_check[1]]

    dist_between_mic = abs(MIC_LOCATION_ARRAY[mic_idx_to_check[1]] - MIC_LOCATION_ARRAY[mic_idx_to_check[0]])

    #doa(signal1, signal2, dist_between_mic, samplerate)

if __name__ == '__main__':

    main()