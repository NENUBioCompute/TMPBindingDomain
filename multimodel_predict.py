import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import argparse
import data_prepare
from pointnet import *
# from pointnet2_cls_msg import *
import tqdm
import data_handler

class my_dataset(Dataset):
    def __init__(self, dataset_path):
        super(my_dataset, self).__init__()
        self.x, self.y = data_prepare.get_dataset(dataset_path)
        self.leng = len(self.y)

    def __getitem__(self, index):
        return {'site_coord': self.x['site_coord'][index], 'site_name': self.x['site_name'][index]}, self.y[index]

    def __len__(self):
        return self.leng


def dataset(dataset_path):
    dataset = my_dataset(dataset_path)
    datas = DataLoader(dataset=dataset, batch_size=32, shuffle=False, pin_memory=True)
    return datas


def evaluate(net, test_loader, no, dev):
    net.eval()
    total_correct = 0
    count = 0
    f1 = open('valid_pred_' + str(no) + '.txt', 'a+')
    f2 = open('valid_label_' + str(no) + '.txt', 'a+')
    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq:
                # label = label[:,0]
                num_examples = label.shape[0]
                data['site_coord'] = data['site_coord'].to(dev)
                data['site_name'] = data['site_name'].to(dev)
                label = label.to(dev)
                logits = net(data)
                logits = logits.squeeze(dim=1)
                # _, preds = logits.max(1)
                preds = logits.ge(0.7).float()
                # preds = torch.round(logits).float()
                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples
                pred_list = preds.cpu().numpy().tolist()
                lable_list = label.cpu().numpy().tolist()
                for i in pred_list:
                    f1.write(str(int(i)) + '\n')
                for i in lable_list:
                    f2.write(str(int(i)) + '\n')
                tq.set_postfix({
                    'AvgAcc': '%.5f' % (total_correct / count)})
    f1.close()
    f2.close()

if __name__ == '__main__':
    test_dataset = dataset(r'./dataset/valid_set.pickle')
    device_list = [0, 1, 2, 3]
    torch.cuda.set_device(7)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    for no in range(29, 30):
        # training_dataset = dataset(r'./dataset/valid_set.pickle')
        net = Pointnet()
        net = net.to(device)
        net.load_state_dict(torch.load('model_' + str(no) + '.pth'), strict=True)
        # opt.load_state_dict(torch.load('optimizer.pt'))
        evaluate(net, test_dataset, no, device)