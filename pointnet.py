import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# from torchsummary import summary

class Pointnet(nn.Module):

    def __init__(self):
        super(Pointnet, self).__init__()
        self.transform_net = transform_net(25)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.feature_transform_net = feature_transform_net(64)
        self.res_name_feature_transform_net = res_name_feature_transform_net(20)
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, point_cloud_dict):
        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']
        coord_net = self.transform_net(point_cloud_coord)
        coord_net = coord_net.permute(0, 2, 1)
        # x = net.unsqueeze(3)
        coord_net = F.relu(self.bn1(self.conv1(coord_net)))
        coord_net = F.relu(self.bn2(self.conv2(coord_net)))
        coord_net = coord_net.permute(0, 2, 1)
        coord_net = self.feature_transform_net(coord_net)
        coord_net = coord_net.permute(0, 2, 1)
        coord_net = F.relu(self.bn3(self.conv3(coord_net)))
        coord_net = F.relu(self.bn4(self.conv4(coord_net)))
        coord_net = F.relu(self.bn5(self.conv5(coord_net)))
        coord_net = torch.max(coord_net, 2, keepdim=True)[0]
        # print(x.size())
        coord_net = coord_net.view(-1, 1024)
        name_net = self.res_name_feature_transform_net(point_cloud_name)
        # print(coord_net.size())
        # print(name_net.size())
        global_freature = torch.cat([coord_net, name_net], axis=-1)
        # print(global_freature.size())
        global_freature = F.relu(self.fc1(global_freature))
        global_freature = torch.sigmoid(self.fc2(global_freature))

        return global_freature

class Pointnet_pretraining_coord(nn.Module):
    def __init__(self):
        super(Pointnet_pretraining_coord, self).__init__()
        self.transform_net = transform_net(25)
        self.feature_transform_net = feature_transform_net(64)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, point_cloud_dict):
        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']
        coord_net = self.transform_net(point_cloud_coord)
        coord_net = coord_net.permute(0, 2, 1)
        # x = net.unsqueeze(3)
        coord_net = F.relu(self.bn1(self.conv1(coord_net)))
        coord_net = F.relu(self.bn2(self.conv2(coord_net)))
        coord_net = coord_net.permute(0, 2, 1)
        coord_net = self.feature_transform_net(coord_net)
        coord_net = coord_net.permute(0, 2, 1)
        coord_net = F.relu(self.bn3(self.conv3(coord_net)))
        coord_net = F.relu(self.bn4(self.conv4(coord_net)))
        coord_net = F.relu(self.bn5(self.conv5(coord_net)))
        coord_net = torch.max(coord_net, 2, keepdim=True)[0]
        # print(x.size())
        coord_net = coord_net.view(-1, 1024)
        # print(x.size())
        coord_net = self.fc3(coord_net)

        coord_net = torch.sigmoid(coord_net)
        return coord_net

class Pointnet_pretraining_name(nn.Module):
    def __init__(self):
        super(Pointnet_pretraining_name, self).__init__()
        self.res_name_feature_transform_net = res_name_feature_transform_net(20)
        self.fc = nn.Linear(256, 1)
        # self.relu = nn.ReLU()

    def forward(self, point_cloud_dict):
        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']
        name_net = self.res_name_feature_transform_net(point_cloud_name)
        name_net = self.fc(name_net)
        return name_net


class transform_net(nn.Module):
    def __init__(self, num_point):
        super(transform_net, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, net):
        batchsize = net.size()[0]
        x = net.permute(0, 2, 1)
        # x = net.unsqueeze(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        # print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        # print(x.size())
        x = torch.matmul(net, x)
        # print(x.size())
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

if __name__ == '__main__':
    pointcloud1 = torch.zeros((32, 25, 3))
    pointcloud2 = torch.zeros((32, 25, 20))
    input = {
        'site_coord': pointcloud1,
        'site_name': pointcloud2
    }
    model = Pointnet()
    print(model(input))


