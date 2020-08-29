import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
from    tensorflow.keras import regularizers
import os

# 结构域识别模型网络构建
class Pointnet(keras.Model):

    def __init__(self):
        super(Pointnet, self).__init__()
        self.transform_net = transform_net(25)
        self.conv1 = layers.Conv2D(64, [1, 3], strides=[1, 1], padding='VALID')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn2 = layers.BatchNormalization()
        self.feature_transform_net = feature_transform_net(25, 64)
        self.conv3 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(128, [1, 1], strides=[1, 1], padding='VALID')
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='VALID')
        self.bn5 = layers.BatchNormalization()
        self.maxpool2d = layers.MaxPooling2D([25, 1], [2, 2], padding='VALID')

        # self.fc3 = layers.Dense(1, activation='sigmoid')

        self.res_name_feature_transform_net = res_name_feature_transform_net(25, 20)
        # self.fc = layers.Dense(1, activation='sigmoid')

        self.full_connect1 = layers.Dense(512, activation='relu')
        self.full_connect2 = layers.Dense(1, activation='sigmoid')



    def call(self, point_cloud_dict):

        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']

        # print('transform')

        coord_net = self.transform_net(point_cloud_coord)
        # print('transform over')

        coord_net = tf.expand_dims(coord_net, -1)
        # print(net.shape)
        coord_net = self.conv1(coord_net)
        # print(net.shape)
        coord_net = self.bn1(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv2(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn2(coord_net)
        coord_net = self.relu(coord_net)
        # print('feature')
        coord_net = self.feature_transform_net(coord_net)
        # print('feature over')
        coord_net = self.conv3(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn3(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv4(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn4(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv5(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn5(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.maxpool2d(coord_net)
        # print(coord_net.shape)
        coord_net = layers.Flatten()(coord_net)
        # print(coord_net.shape)

        # coord_net = self.fc3(coord_net)
        # return coord_net

        # name_net = self.res_name_feature_transform_net(point_cloud_name)
        # name_net = self.fc(name_net)
        # # print(name_net.shape)
        # return name_net

        name_net = self.res_name_feature_transform_net(point_cloud_name)
        global_freature = tf.concat([coord_net, name_net], axis=-1)
        # print(global_freature.shape)
        global_freature = self.full_connect1(global_freature)
        global_freature = self.full_connect2(global_freature)

        return global_freature


class Pointnet_pretraining_coord(keras.Model):

    def __init__(self):
        super(Pointnet_pretraining_coord, self).__init__()
        self.transform_net = transform_net(25)
        self.conv1 = layers.Conv2D(64, [1, 3], strides=[1, 1], padding='VALID')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn2 = layers.BatchNormalization()
        self.feature_transform_net = feature_transform_net(25, 64)
        self.conv3 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(128, [1, 1], strides=[1, 1], padding='VALID')
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='VALID')
        self.bn5 = layers.BatchNormalization()
        self.maxpool2d = layers.MaxPooling2D([25, 1], [2, 2], padding='VALID')

        self.fc3 = layers.Dense(1, activation='sigmoid')





    def call(self, point_cloud_dict):

        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']

        # print('transform')

        coord_net = self.transform_net(point_cloud_coord)
        # print('transform over')

        coord_net = tf.expand_dims(coord_net, -1)
        # print(net.shape)
        coord_net = self.conv1(coord_net)
        # print(net.shape)
        coord_net = self.bn1(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv2(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn2(coord_net)
        coord_net = self.relu(coord_net)
        # print('feature')
        coord_net = self.feature_transform_net(coord_net)
        # print('feature over')
        coord_net = self.conv3(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn3(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv4(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn4(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.conv5(coord_net)
        # print(coord_net.shape)
        coord_net = self.bn5(coord_net)
        coord_net = self.relu(coord_net)
        coord_net = self.maxpool2d(coord_net)
        # print(coord_net.shape)
        coord_net = layers.Flatten()(coord_net)
        # print(coord_net.shape)

        coord_net = self.fc3(coord_net)
        return coord_net


class Pointnet_pretraining_name(keras.Model):

    def __init__(self):
        super(Pointnet_pretraining_name, self).__init__()

        self.res_name_feature_transform_net = res_name_feature_transform_net(25, 20)
        self.fc = layers.Dense(1, activation='sigmoid')

    def call(self, point_cloud_dict):

        point_cloud_coord = point_cloud_dict['site_coord']
        point_cloud_name = point_cloud_dict['site_name']

        name_net = self.res_name_feature_transform_net(point_cloud_name)
        name_net = self.fc(name_net)
        # print(name_net.shape)
        return name_net


# STN空间变换矩阵（针对N x 3坐标矩阵）
class transform_net(layers.Layer):
    def __init__(self, num_point):
        super(transform_net, self).__init__()

        self.conv1 = layers.Conv2D(64, [1, 3], strides=[1, 1], padding='VALID')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(128, [1, 1], strides=[1, 1], padding='VALID')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='VALID')
        self.bn3 = layers.BatchNormalization()
        self.maxpool2d = layers.MaxPooling2D([num_point, 1], [2, 2], padding='VALID')
        self.fc1 = layers.Dense(512, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.fc2 = layers.Dense(256, activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.kernel = self.add_weight('w', [256, 9])
        self.bias = self.add_weight('b', [9])

    def call(self, pointcloud):
        input = tf.expand_dims(pointcloud, -1)
        batch_size = input.get_shape()[0]
        # print(input.shape)
        net = self.conv1(input)
        # print(net.shape)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.conv2(net)
        # print(net.shape)
        net = self.bn2(net)
        net = self.relu(net)
        net = self.conv3(net)
        # print(net.shape)
        net = self.bn3(net)
        net = self.relu(net)
        net = self.maxpool2d(net)
        # print(net.shape)
        # net = tf.reshape(net, [batch_size, 1024])
        net = layers.Flatten()(net)
        # print(net.shape)
        net = self.fc1(net)
        # print(net.shape)
        net = self.bn4(net)
        net = self.fc2(net)
        # print(net.shape)
        net = self.bn5(net)
        net = net @ self.kernel + self.bias
        # print(net.shape)
        net = tf.reshape(net, [-1, 3, 3])
        # print(net.shape)
        net = tf.matmul(pointcloud, net)
        # print(net.shape)
        return net


# STN空间变换矩阵（针对N x D特征矩阵）
class feature_transform_net(layers.Layer):
    def __init__(self, num_point, k=64):
        super(feature_transform_net, self).__init__()
        self.k = k
        self.conv1 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(128, [1, 1], strides=[1, 1], padding='VALID')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='VALID')
        self.bn3 = layers.BatchNormalization()
        self.maxpool2d = layers.MaxPooling2D([num_point, 1], [2, 2], padding='VALID')
        self.fc1 = layers.Dense(512, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.fc2 = layers.Dense(256, activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.kernel = self.add_weight('w', [256, k*k])
        self.bias = self.add_weight('b', [k*k])

    def call(self, pointcloud):

        net = self.conv1(pointcloud)
        # print(net.shape)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.conv2(net)
        # print(net.shape)
        net = self.bn2(net)
        net = self.relu(net)
        net = self.conv3(net)
        # print(net.shape)
        net = self.bn3(net)
        net = self.relu(net)
        net = self.maxpool2d(net)
        # print(net.shape)
        net = layers.Flatten()(net)
        # print(net.shape)
        net = self.fc1(net)
        # print(net.shape)
        net = self.bn4(net)
        net = self.fc2(net)
        # print(net.shape)
        net = self.bn5(net)
        net = net @ self.kernel + self.bias
        # print(net.shape)
        net = tf.reshape(net, [-1, self.k, self.k])
        # print(net.shape)
        net = tf.matmul(tf.squeeze(pointcloud, axis=[2]), net)
        # print(net.shape)
        net = tf.expand_dims(net, [2])
        # print(net.shape)
        return net


class res_name_feature_transform_net(layers.Layer):
    def __init__(self, num_point, k=25):
        super(res_name_feature_transform_net, self).__init__()
        self.k = k
        self.name_feature_transform_net = feature_transform_net(25, 20)
        self.conv1 = layers.Conv2D(64, [1, 1], strides=[1, 1], padding='VALID')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(256, [1, 1], strides=[1, 1], padding='VALID')
        self.bn2 = layers.BatchNormalization()
        self.maxpool2d = layers.MaxPooling2D([num_point, 1], [2, 2], padding='VALID')

    def call(self, pointnet):
        pointnet = tf.expand_dims(pointnet, -2)
        net = self.name_feature_transform_net(pointnet)
        # print('trans')
        # print(net.shape)
        # print('pointnet')
        # print(pointnet.shape)
        net = self.conv1(net)
        net = self.bn1(net)
        net = self.relu(net)
        # print(net.shape)
        net = self.conv2(net)
        net = self.bn2(net)
        net = self.relu(net)
        # print(net.shape)
        net = self.maxpool2d(net)
        # print(net.shape)
        net = layers.Flatten()(net)
        # print(net.shape)
        return net


if __name__ == '__main__':
    pointcloud1 = tf.fill([32, 25, 3], 0)
    pointcloud1 = tf.cast(pointcloud1, dtype=tf.float32)
    pointcloud2 = tf.fill([32, 25, 20], 0)
    pointcloud2 = tf.cast(pointcloud2, dtype=tf.float32)
    input = {
        'site_coord': pointcloud1,
        'site_name': pointcloud2
    }

    model = Pointnet_pretraining_name()
    model(input)
    model.summary()
