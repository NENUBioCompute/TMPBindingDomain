PDB文件夹：对于PDB文件及数据的处理，因为经过多次修改比较乱，后续会整理代码
bagging.py：用于集成所有模型的结果，还没有完善
bagging_models_training.py：训练模型，目前暂定训练30个
clstr_parse.py：解析clstr
data_prepare.py：数据集整理与处理
evaluate.py：评估模型
pointnet.py：预训练与训练的模型构建
predict.py：预测单个pdb

dataset为一个list，list中每一项为一个样本，内容为[ [坐标矩阵, 氨基酸种类矩阵] , label]

直接运行bagging_models_traning就可以开始训练模型，可以修改训练参数并增加多进程提高效率。