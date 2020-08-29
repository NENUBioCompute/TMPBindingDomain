import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
from    tensorflow.keras import regularizers
import 	os
import 	pointnet
import 	data_prepare
import 	time
import  evaluate


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


def dataset(dataset_path):
    x, y = data_prepare.get_dataset(dataset_path)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(10000).batch(32)
    return dataset


# 训练参数设置
def training_parameter(model_name):
    TensorBoardcallback = keras.callbacks.TensorBoard(
        log_dir=r'logs\{}'.format(model_name),
        histogram_freq=1, batch_size=32,
        write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch'
    )

    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=r'model\\' + model_name + '.h5',
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        period=1
    )
    return TensorBoardcallback, ModelCheckpoint


# 预训练
def pretraining(model_id, training_dataset, test_dataset, valid_dataset):
    # 坐标预训练模型
    TensorBoardcallback, ModelCheckpoint = training_parameter('pretrain_coord_{}'.format(model_id))
    pretraining_coord_model = pointnet.Pointnet_pretraining_coord()
    pretraining_coord_model.compile(
        optimizer=optimizers.Adam(lr=0.00001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    pretraining_coord_model.fit(training_dataset, epochs=1000, validation_data=valid_dataset, callbacks=[TensorBoardcallback, ModelCheckpoint])
    # 氨基酸种类预训练模型
    TensorBoardcallback, ModelCheckpoint = training_parameter('pretrain_name_{}'.format(model_id))
    pretraining_name_model = pointnet.Pointnet_pretraining_name()
    pretraining_name_model.compile(
        optimizer=optimizers.Adam(lr=0.00001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    pretraining_name_model.fit(training_dataset, epochs=1000, validation_data=valid_dataset, callbacks=[TensorBoardcallback, ModelCheckpoint])


# 训练模型
def training(model_id, training_dataset, test_dataset, valid_dataset):
    # 正式训练模型
    TensorBoardcallback, ModelCheckpoint = training_parameter('bagging_model_{}'.format(model_id))
    bagging_model = pointnet.Pointnet()
    bagging_model.compile(
        optimizer=optimizers.Adam(lr=0.0001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    # 初始化输入
    pointcloud1 = tf.fill([32, 25, 3], 0)
    pointcloud1 = tf.cast(pointcloud1, dtype=tf.float32)
    pointcloud2 = tf.fill([32, 25, 20], 0)
    pointcloud2 = tf.cast(pointcloud2, dtype=tf.float32)
    input = {
        'site_coord': pointcloud1,
        'site_name': pointcloud2
    }
    bagging_model(input)
    # bagging_model.fit(training_dataset, epochs=0, validation_data=valid_dataset, callbacks=[TensorBoardcallback, ModelCheckpoint])
    bagging_model.summary()

    bagging_model.load_weights(r'model\pretrain_coord_{}.h5'.format(model_id), by_name=True)
    bagging_model.load_weights(r'model\pretrain_name_{}.h5'.format(model_id), by_name=True)
    bagging_model.fit(training_dataset, epochs=1000, validation_data=valid_dataset, callbacks=[TensorBoardcallback, ModelCheckpoint])


if __name__ == '__main__':
    test_dataset = dataset(r'dataset\test_set.pickle')
    valid_dataset = dataset(r'dataset\valid_set.pickle')
    print('start')
    for i in range(30):
        training_dataset = dataset(r'dataset\bagging_set{}.pickle'.format(i))
        pretraining(i, training_dataset, test_dataset, valid_dataset)
        break
        training(i, training_dataset, test_dataset, valid_dataset)