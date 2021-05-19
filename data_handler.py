# -*- coding: utf-8 -*-
"""
    Spyder Editor
    Author: YihangBao
    Created: 14/04/2020
"""
import math


def judge(standard_re, result, file_path):
    right_pre = 0
    f = open(file_path, 'a+')
    for i in range(367831):
        if standard_re[i] == result[i]:
            right_pre += 1
    f.write('ACC: ' + str(right_pre / 367831) + '\n')
    
    fn = 0
    fp = 0
    tp = 0
    tn = 0
    wh = 1
    for i in range(367831):
        if standard_re[i] != wh and result[i] != wh:
            tn += 1
        elif standard_re[i] != wh and result[i] == wh:
            fp += 1
        elif standard_re[i] == wh and result[i] == wh:
            tp += 1
        elif standard_re[i] == wh and result[i] != wh:
            fn += 1

    r = tp / (tp + fn)
    p = tp / (tp + fp)

    f.write('precision: ' + str(p) + '\n')
    f.write('recall: ' + str(r) + '\n')
    f.write('speci:' + str(tn / (tn + fp)) + '\n')
    f.write('mcc:' + str((tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))) + '\n')
    f.write('f-measure:' + str((2 * p * r) / (p + r)) + '\n')
    f.close()
    return p
