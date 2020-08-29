from PDB.DataProcess.site_info import *
import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
from    tensorflow.keras import regularizers
import 	os
import 	pointnet
import 	data_prepare
import  evaluate
import 	time
from collections import Counter



def get_model(batch_size):
    pointcloud1_init = tf.fill([batch_size, 25, 3], 0)
    pointcloud1_init = tf.cast(pointcloud1_init, dtype=tf.float32)
    pointcloud2_init = tf.fill([batch_size, 25, 20], 0)
    pointcloud2_init = tf.cast(pointcloud2_init, dtype=tf.float32)
    input = {
        'site_coord': pointcloud1_init,
        'site_name': pointcloud2_init
    }
    model = pointnet.Pointnet()
    model(input)
    model.load_weights(r'H:\biodata\ml\biodata_pointnet\model\domain_model-1590849598.h5', by_name=True)
    return model


def printinfo(pred, y):
    evaluator = evaluate.evaluate()
    evaluator.roc_curve(pred, y)
    evaluator.pr_curve(pred, y)
    evaluator.evaluate(pred, y, 0.5)
    print('fp = {}'.format(evaluator.fp))
    print('tp = {}'.format(evaluator.tp))
    print('fn = {}'.format(evaluator.fn))
    print('tn = {}'.format(evaluator.tn))
    print('precision = {}'.format(evaluator.precision))
    print('recall = {}'.format(evaluator.recall))
    print('mcc = {}'.format(evaluator.mcc))
    print('acc = {}'.format(evaluator.acc))
    print('f1 = {}'.format(evaluator.f_score(1)))


def predict(model, x, y):
    model.compile(
        optimizer=optimizers.Adam(lr=0.0001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    input = tf.data.Dataset.from_tensor_slices(x)
    input = input.batch(32)
    pred = model.predict(x=input)
    printinfo(pred, y)

def _data_to_dataset(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    # print(len(data))
    site_coord_list = []
    site_name_list = []
    site_ispositive_list = []
    for site in data:
        res_coord_list = []
        res_name_list = []
        for res in site[0]:
            res_coord_list.append(res[0])
            res_name_list.append(res[1])
        # print(res_coord_list)
        # print(res_name_list)
        site_coord_list.append(res_coord_list)
        site_name_list.append(res_name_list)
        site_ispositive_list.append(site[1])
        # if len(site_ispositive_list) == 10:
        #     break
    site_name_list = tf.one_hot(site_name_list, depth=20)
    input = {
        'site_coord': site_coord_list,
        'site_name': site_name_list
    }
    ds = tf.data.Dataset.from_tensor_slices((input, site_ispositive_list))
    ds = ds.shuffle(10000).batch(32)
    # ds = ds.repeat()
    return ds

if __name__ == '__main__':
    x, y = data_prepare.get_dataset(r'H:\biodata\pdbtm\dataset_3582442\1a0t.pdb.pickle')
    model = get_model(32)
    predict(model, x, y)