import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

class Pointnet(nn.Module):

    def __init__(self):
        super(Pointnet, self).__init__()
        self.res_name_feature_transform_net = res_name_feature_transform_net(20)
        self.pointnetpp = EnsembleRCNet(1,'cpu')
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, point_cloud_dict):
        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']
        coord_net = point_cloud_coord.permute(0,2,1)
        coord_net = self.pointnetpp(coord_net)
        name_net = self.res_name_feature_transform_net(point_cloud_name)
        # print(coord_net.size())
        # print(name_net.size())
        global_freature = torch.cat([coord_net, name_net], axis=-1)
        # print(global_freature.size())
        global_freature = F.relu(self.fc1(global_freature))
        global_freature = torch.sigmoid(self.fc2(global_freature))

        return global_freature

class res_name_feature_transform_net(nn.Module):
    def __init__(self, k=25):
        super(res_name_feature_transform_net, self).__init__()
        self.name_feature_transform_net = feature_transform_net(20)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, net):
        batchsize = net.size()[0]
        x = self.name_feature_transform_net(net)
        x = net.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # print(x.size())
        x = x.view(-1, 256)
        # print(x.size())
        return x

def prepare_input_first_level(tensor_x, tensor_quantiles):
    x = tensor_x.numpy()
    quantiles = tensor_quantiles.numpy()

    base = 0
    batch_num = x.shape[0]

    stack_quantiles = quantiles.reshape(-1)
    nonzero_quantiles = stack_quantiles[np.nonzero(stack_quantiles)]  # get the nonzero entries
    max_len = np.max(nonzero_quantiles)
    num_seq = nonzero_quantiles.size

    res_seq = np.zeros((num_seq, max_len, 3))

    stack_data = x.reshape(-1, 3)
    for i in range(num_seq):
        num = nonzero_quantiles[i]
        res_seq[i, 0:num, :] = stack_data[base:base + num, :]
        base = base + num

    # sort the data by sequence length
    sort_index = np.argsort(nonzero_quantiles)[::-1]  # sort in descending order according to length
    inverse_index = np.argsort(sort_index)

    sort_data = res_seq[sort_index, :, :]
    sort_len = nonzero_quantiles[sort_index]

    return sort_data, sort_len, inverse_index


class RNNUnit(nn.Module):
    def __init__(self, hidden_size, device):
        super(RNNUnit, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size,
                          num_layers=2, bidirectional=False, batch_first=True).to(device)
        self.device = device

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices):
        packed_seq_data = pack_padded_sequence(seq_data, seq_len, batch_first=True)
        packed_output, _ = self.rnn(packed_seq_data)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the last time step for each sequence
        seq_len = torch.LongTensor(seq_len)
        idx = (seq_len - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
        decoded = output.gather(1, idx.to(self.device)).squeeze(1)

        # unsort the input sequence
        inverse_index = torch.LongTensor(inverse_index)
        odx = inverse_index.view(-1, 1).expand(seq_len.size(0), output.size(-1))
        decoded = decoded.gather(0, odx.to(self.device))

        # finally, prepare for the data for following convolution
        batch_num = quantiles.shape[0]
        plane_num = quantiles.shape[1]

        res_image = torch.zeros(batch_num * plane_num * plane_num, output.size(2)).to(self.device)

        num_items = items_indices.shape[0]
        items_indices = torch.LongTensor(items_indices).view(-1, 1).expand(num_items, output.size(2))
        res_image = res_image.scatter_(0, items_indices.to(self.device), decoded)

        res_image = res_image.view(batch_num, plane_num, plane_num, output.size(2))
        res_image = res_image.permute(0, 3, 1, 2).contiguous()

        return res_image


class RCNet(nn.Module):
    def __init__(self, device):
        super(RCNet, self).__init__()
        self.rnn = RNNUnit(hidden_size=64, device=device)

        self.conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=2, bias=False)

        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn0 = nn.BatchNorm2d(64)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(256)

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices):
        res_image = self.rnn(x, quantiles, seq_data, seq_len, inverse_index, items_indices)
        res_image = self.bn0(res_image)

        x = F.relu(self.bn1(self.conv1(res_image)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))  # Seems like the existence of this 6-th conv layer has no effect on the final classification accuracy...

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 8192)
        x = F.dropout(x)
        x = F.relu(self.fc_bn1(self.fc1(x)))

        return F.log_softmax(x, dim=1)


class EnsembleRCNet(nn.Module):
    def __init__(self, which_dir, device):
        super(EnsembleRCNet, self).__init__()
        self.which_dir = which_dir

        if which_dir == 1:
            self.RCNet1 = RCNet(device)
        elif which_dir == 2:
            self.RCNet2 = RCNet(device)
        else:
            self.RCNet3 = RCNet(device)

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices):
        if self.which_dir == 1:
            x = self.RCNet1(x, quantiles, seq_data, seq_len, inverse_index, items_indices)
        elif self.which_dir == 2:
            x = self.RCNet2(x, quantiles, seq_data, seq_len, inverse_index, items_indices)
        else:
            x = self.RCNet3(x, quantiles, seq_data, seq_len, inverse_index, items_indices)

        return x


class feature_transform_net(nn.Module):
    def __init__(self, k=64):
        super(feature_transform_net, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, net):
        batchsize = net.size()[0]
        x = net.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # print(x.size())
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        x = torch.matmul(net, x)
        return x

if __name__ == '__main__':
    pointcloud1 = torch.zeros((32, 25, 3))
    pointcloud2 = torch.zeros((32, 25, 20))
    input = {
        'site_coord': pointcloud1,
        'site_name': pointcloud2
    }
    model = Pointnet()
    model(input)