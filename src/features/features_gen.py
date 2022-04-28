import preprocessing
import RIR_Gen
import numpy as np
import soundfile as sf
import os
import csv
import pandas as pd
import gcc

def read_mics_position(data_path):

    """
    read scv file of mics positions, and return the numpy array of the positions
    """

    mics_position = []

    with open(data_path, 'r', newline='') as f:

        line_read = csv.reader(f, delimiter=',')
        next(line_read) #first line is header

        # loop over all the lines, convert the string to float, and append to the list
        for row in line_read:
            mics_position.append([float(i) for i in row])
    
    mics_position = np.array(mics_position)

    return mics_position

def write_feature(data,data_path, number_of_direction, type):
    """
    reshape the feature to 2D instead of 3D (flat the 2 last dimension)
    and write the features to csv file, with headers of each of the dimension
    """    

    num_of_frames = data.shape[0]
    fearure_size_flat = data.shape[1] * data.shape[2]
    list_of_direction = np.linspace(0, 360, number_of_direction, endpoint=False)

    #create the headers of the columns
    column1 = [''] * fearure_size_flat
    if type == "time":
        column1[::data.shape[1]] = ['mic_num_' + str(i) for i in range(1,data.shape[2]+1)]
    elif type == "correlation" or type == "spectrum": 
        column1[::data.shape[1]] = ['direction: ' + str(i) + '[deg]' for i in list_of_direction]
   
    column2 = range(fearure_size_flat)
    if type == "spectrum": 
        column2 = ['frequency ' + str(i) for i in range(fearure_size_flat)]

    #create the headers of the rows
    index_list = ['frame_' + str(i) for i in range(1, num_of_frames + 1)]

    # reshape the date to flat
    data_flat = np.reshape(data, (data.shape[0],-1))
        
    df = pd.DataFrame(
        data_flat , 
        index= index_list,
        columns=pd.MultiIndex.from_arrays([column1, column2])
        )
    df.to_csv(data_path, index=True)

def read_VAD_lables(VAD_file_path):

    VAD_lables = []

    with open(VAD_file_path, 'r', newline='') as f:

        line_read = csv.reader(f, delimiter=',')

        # loop over all the lines, convert the string to float, and append to the list
        for row in line_read:
            VAD_lables = [float(i) for i in row]
    
    VAD_lables = np.array(VAD_lables)

    return VAD_lables

def write_lables(VAD_lables, lables_file_path, frame_size, overlap):

    lables = []

    length = len(VAD_lables)
    i=0
    while i < length - frame_size:
        block = VAD_lables[i:i+frame_size]

        # count number of -1 in block
        x = np.count_nonzero(block==-1)

        if x > frame_size/2:
            lables.append(-1)
        else:
            lables.append(block[block != -1][0])

        i = i + frame_size - overlap 

    df = pd.DataFrame(lables)
    df.to_csv(lables_file_path, index=True)

def features_gen(data_folder, features_folder):

    number_of_direction = 36
    frame_size = 1024
    overlap = 512

    # create the directories, if they aren't exist
    RIR_Gen.create_dirs([features_folder, features_folder + 'preprocessing1',features_folder + 'preprocessing2', features_folder + 'preprocessing3'])

    # loop over all the files in the data folder
    files = os.listdir(data_folder + 'mics/random_array')

    for file in files:

        print("generate features of tile: " + file)

        record_name = file.rsplit('.',1)[0]

        signal, samplerate = sf.read(data_folder + 'mics/random_array/' + file)
        signal_const, _ = sf.read(data_folder + 'mics/fix_array/' + record_name + '.wav')

        mics_position = read_mics_position(data_folder + 'mics_position/random_array/' + record_name + '.csv')
        mics_position_const = read_mics_position(data_folder + 'mics_position/fix_array/' + record_name + '.csv')

        # preprocessing 1 - time
        time_pp_matrix = preprocessing.signal_preprocessing(
            signal_const, mics_position_const, number_of_direction, samplerate, frame_size, overlap, type = "time"
            )
        # write the data to csv file
        # reshape the data: from 3D: (num_of_frames, frame_size, number_of_mic)
        # to 2D (num_of_frames, frame_size * number_of_mic)
        write_feature(time_pp_matrix,features_folder + 'preprocessing1/' + record_name + '.csv', number_of_direction, type = "time")

        # preprocessing 2 - correlation
        correlation_pp_matrix = preprocessing.signal_preprocessing(
            signal, mics_position, number_of_direction, samplerate, frame_size, overlap, type = "correlation"
            )
        # reshape the data: from 3D: (num_of_frames, 1, number_of_direction)
        # to 2D (num_of_frames, 1 * number_of_direction)
        write_feature(correlation_pp_matrix,features_folder + 'preprocessing2/' + record_name + '.csv', number_of_direction, type = "correlation")
        
        # preprocessing 3 - spectrum
        spectrum_pp_matrix = preprocessing.signal_preprocessing(
            signal, mics_position, number_of_direction, samplerate, frame_size, overlap, type = "spectrum"
            )
        # reshape the data: from 3D: (num_of_frames, num_of_frequency, number_of_direction)
        # to 2D (num_of_frames, num_of_frequency * number_of_direction)
        write_feature(spectrum_pp_matrix,features_folder + 'preprocessing3/' + record_name + '.csv', number_of_direction, type = "spectrum")

        # create lables
        VAD_lables = read_VAD_lables(data_folder + 'VAD_lables/random_array/' + record_name + '.csv')
        write_lables(VAD_lables, features_folder + 'lables/' + record_name + '.csv', frame_size, overlap)
        
def main():

    features_gen("Final_Proj/input/data/", "Final_Proj/input/features/")

if __name__ == '__main__':

    main()