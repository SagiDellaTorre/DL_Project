import os

def rename_files_by_oreder(path):

    new_index = 0
    file_first = "/file_"
    file_end = ".csv"

    for i in range(3000):
        old_name = path + file_first + str(i) + file_end
        new_name = path + file_first + str(new_index) + file_end
        if os.path.exists(old_name):
            if i != new_index:
                os.rename(old_name, new_name)
            new_index += 1

def main():
    
    # path = "/home/dsi/dellats1/DL_Project/data/features/preprocessing1"
    rename_files_by_oreder(path)

if __name__ == '__main__':

    main()