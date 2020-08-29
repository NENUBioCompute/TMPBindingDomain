from random import shuffle
import csv
f = open('seq_cdhit_40.clstr', 'r')
lines = f.readlines()
cluster = {}
local_cluster = -1
for line in lines:
    str = line.split()
    if str[0] == '>Cluster':
        local_cluster = str[1]
        cluster[local_cluster] = []
    else:
        cluster[local_cluster].append([str[2][1:5]])
print(len(cluster.keys()))
print(cluster)
x = ['%s' % i for i in range(935)]
print(x)
shuffle(x)
print(x)
total = 4098
training_set = []
valid_set = []
test_set = []
for i in x:
    if len(training_set) <= 2800:
        training_set += cluster[i]
    elif len(valid_set) <= 800:
        valid_set += cluster[i]
    else:
        test_set += cluster[i]

print(len(training_set))
print(training_set)
print(len(valid_set))
print(valid_set)
print(len(test_set))
print(test_set)

with open('test_id.txt', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(test_set)

with open('valid_id.txt', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(valid_set)

with open('train_id.txt', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(training_set)