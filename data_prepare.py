import pickle
import os
import PDB.DataProcess.data_prepare
import torch.nn.functional as F
import torch


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
    site_name_list = torch.tensor(site_name_list)
    site_name_list = F.one_hot(site_name_list, 20).float()
    site_coord_list = torch.tensor(site_coord_list).float()
    label_list = torch.tensor(label_list).float()
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

if __name__ == '__main__':
    # training_x, training_y, valid_x, valid_y, test_x, test_y = training_dataset()
    x, y = get_dataset(r'path')
