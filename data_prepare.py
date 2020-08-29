import pickle
import os
import tensorflow as tf
import PDB.DataProcess.data_prepare



def _data_to_site(data_list):
    site_coord_list = []
    site_name_list = []
    label_list = []
    for site in data_list:
        res_coord_list = []
        res_name_list = []
        for res in site[0]:
            res_coord_list.append(res[0])
            res_name_list.append(res[1])
        site_coord_list.append(res_coord_list)
        site_name_list.append(res_name_list)
        label_list.append(site[1])
    site_name_list = tf.one_hot(site_name_list, depth=20)

    return (site_coord_list, site_name_list), label_list


def pickle_to_input(path):
    # data_list = PDB.DataProcess.data_prepare.get_dataset(path)
    f = open(path, 'rb')
    data_list = pickle.load(f)
    f.close()
    (site_coord_list, site_name_list), label_list = _data_to_site(data_list)
    input = {
        'site_coord': site_coord_list,
        'site_name': site_name_list
    }
    return input, label_list


def get_dataset(path):
    x, y = pickle_to_input(path)
    return x, y



def get_filelist(dir, Filelist):
    """get the file list under the direction"""
    newDir = dir
    if os.path.isfile(dir):
        suffix = os.path.splitext(dir)[1]
        if suffix == ".pickle":
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

# def get_site_pickle():
#     file_list = []
#     file_list = get_filelist(r'H:\biodata\pdbtm\dataset_3582442', file_list)
#     for file in file_list:
#         filepath, filename = os.path.split(file)
#         new_path = r'H:\biodata\pdbtm\site2\{}'.format(filename)
#         print(new_path)
#         data_list = PDB.DataProcess.data_prepare.get_dataset(file)
#         (site_coord_list, site_name_list), label_list = _data_to_site(data_list)
#         f = open(new_path, 'wb')
#         pickle.dump(((site_coord_list, site_name_list), label_list), f)
#         f.close()

# def get_site_pickle():
#     file_list = []
#     file_list = get_filelist(r'H:\biodata\pdbtm\dataset_3582442', file_list)
#     for file in file_list:
#         filepath, filename = os.path.split(file)
#         new_path = r'H:\biodata\pdbtm\site2\{}'.format(filename)
#         print(new_path)
#         data_list = PDB.DataProcess.data_prepare.get_dataset(file)
#         f = open(new_path, 'wb')
#         pickle.dump(data_list, f)
#         f.close()




if __name__ == '__main__':
    # training_x, training_y, valid_x, valid_y, test_x, test_y = training_dataset()
    x, y = get_dataset(r'path')
