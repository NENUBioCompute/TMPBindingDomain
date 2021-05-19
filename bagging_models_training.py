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
    datas = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    return datas


def train(net, opt, scheduler, train_loader, dev):
    net.train()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            # label = label[:, 0]
            num_examples = label.shape[0]
            data['site_coord'] = data['site_coord'].to(dev)
            data['site_name'] = data['site_name'].to(dev)
            label = label.to(dev)
            opt.zero_grad()
            logits = net(data)
            # loss = nn.CrossEntropyLoss(logits, label)
            Loss = nn.BCELoss()
            logits = logits.squeeze(dim=1)
            loss = Loss(logits, label)
            loss.backward()
            opt.step()
            # printinfo(logits, label.int())
            # break
            preds = torch.round(logits).float()
            # _, preds = logits.max(1)
            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct
            tq.set_postfix({
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AvgAcc': '%.5f' % (total_correct / count)})

    scheduler.step()

def evaluate(net, test_loader, file_path, dev):
    net.eval()
    total_label = []
    total_pred = []
    total_correct = 0
    count = 0
    # f1 = open('pred.txt', 'a+')
    # f2 = open('label.txt','a+')
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
                    total_pred.append(int(i))
                for i in lable_list:
                    total_label.append(int(i))
                # for i in pred_list:
                #     f1.write(str(int(i)) + '\n')
                # for i in lable_list:
                #     f2.write(str(int(i)) + '\n')
                tq.set_postfix({
                    'AvgAcc': '%.5f' % (total_correct / count)})
    pre = data_handler.judge(total_label, total_pred, file_path)
    # f1.close()
    # f2.close()
    return pre

if __name__ == '__main__':
    device_list = [0, 1, 2, 3]
    torch.cuda.set_device(7)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    for no in range(28, 29):
        training_dataset = dataset(r'./dataset/bagging_set{}.pickle'.format(no))
        # training_dataset = dataset(r'./dataset/valid_set.pickle')
        net = Pointnet()
        net = net.to(device)
        opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.7)
        test_dataset = dataset(r'./dataset/test_set.pickle')
        best_test_pre = 0
        # net.load_state_dict(torch.load('epoch_9_0.04905059530882388.pth'), strict=True)
        # opt.load_state_dict(torch.load('optimizer.pt'))
        for epoch in range(10):
            train(net, opt, scheduler, training_dataset, device)
            if (epoch + 1) % 5 == 0:
                print('Epoch #%d Testing' % epoch)
                test_pre = evaluate(net, test_dataset, './result/no_' + str(no) + 'epoch_' + str(epoch) + '.txt', device)
                # if test_pre > best_test_pre:
                torch.save(net.state_dict(), f"{no}epoch_{epoch}_{test_pre}.pth")
                best_test_pre = test_pre
                # torch.save(opt, f'optimizer.pt')
                print('Current test pre: %.5f (best: %.5f)' % (test_pre, best_test_pre))
        # torch.save(net.state_dict(), f"add_model_{no}.pth")