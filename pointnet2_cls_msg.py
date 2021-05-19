import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch
import numpy as np
from torch.autograd import Variable

class Pointnet(nn.Module):

    def __init__(self):
        super(Pointnet, self).__init__()
        self.res_name_feature_transform_net = res_name_feature_transform_net(20)
        self.pointnetpp = get_model()
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

class get_model(nn.Module):
    def __init__(self,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [4, 8, 16], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [4, 8, 16], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

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
    print(model(input))

