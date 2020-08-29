import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import 	time


class evaluate:
    def __init__(self):
        self.fn = 0
        self.fp = 0
        self.tp = 0
        self.tn = 0
        self.precision = 0
        self.recall = 0
        self.specificity = 0
        self.mcc = 0
        self.acc = 0

    def evaluate(self, pred, label, thr=0.5):
        fn = 0
        fp = 0
        tp = 0
        tn = 0
        wh = 0
        for i in range(len(label)):
            if label[i] != wh and pred[i] >= thr:
                tp += 1
            elif label[i] != wh and pred[i] < thr:
                fn += 1
            elif label[i] == wh and pred[i] < thr:
                tn += 1
            elif label[i] == wh and pred[i] >= thr:
                fp += 1
        self.fn = fn
        self.fp = fp
        self.tp = tp
        self.tn = tn
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.specificity = tn / (tp + fn)
        self.mcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        self.acc = (tp + tn) / (fp + fn + tp + tn)

    def f_score(self, n):
        f_score = (1 + n**2) * ((self.precision * self.recall) / (n**2 * self.precision + self.recall))
        return f_score

    def roc_curve(self, pred, label):
        fpr, tpr, thresholds = metrics.roc_curve(label, pred)
        auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(18, 6))
        lw = 2
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, 'k--', color='MediumOrchid', lw=lw, label='ROC curve of class H(area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('result/roc_curve_{}.png'.format(int(time.time())))
        plt.show()

    def pr_curve(self, pred, label):
        precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
        print(len(thresholds))
        print('xxx')
        area = metrics.auc(recall, precision)

        plt.figure(figsize=(18, 6))
        lw = 2
        plt.plot(recall, precision, 'k--', color='MediumOrchid', lw=lw, label='Precision-recall curve of class H(area = %0.2f)' % area)
        plt.plot([1, 0], [0, 1], 'k--', color='navy', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve to Multi-class')
        plt.legend(loc="lower left")
        plt.savefig('result/pr_curve_{}.png'.format(int(time.time())))
        plt.show()
