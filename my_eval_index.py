from doctest import OutputChecker
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import os
from math import *
class ConfusionMatrix(object):


    def __init__(self, num_classes: int, labels: list,preds,true_labels):
        # self.matrix = np.zeros((num_classes, num_classes))#初始化混淆矩阵，元素都为0
        self.matrix=confusion_matrix(true_labels, preds)
        self.num_classes = num_classes#类别数量，eye_state为3
        self.labels = labels#类别标签
        self.output_path='output/eval_result/'
        if  not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    # def update(self, preds, labels):
    #     # for p, t in zip(preds, labels):#pred为预测结果，labels为真实标签
    #     #     self.matrix[p, t] += 1#根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1
    #     # matrix=confusion_matrix(labels, preds)

    def summary(self):#计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.matrix.shape[0]):
            sum_TP += self.matrix[i, i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n#总体准确率
        print("the model accuracy is ", acc)
		
		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()#创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity","FPR"]
        for i in range(self.matrix.shape[0]):#精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FN = np.sum(self.matrix[i, :]) - TP
            FP = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            FPR=round(FP / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity,FPR])
            
            with open ('./output/eval_result/'+self.labels[i]+'.txt','a+') as t:
                t.write(str(Precision)+' '+str(Recall)+' '+str(Specificity)+' '+str(FPR)+'\n')
                
        print(table)
        return str(acc),self.labels[i],str(Precision),str(Recall),str(FPR)

    def plot(self):#绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # plt.show()
        plt.savefig('output/matrix2.jpg')
    def sum_index(self):
        matrix = self.matrix
        print(matrix)
        
        for file in os.listdir(self.output_path):
            pre=[]
            rec=[]
            FRP=[]
            with open (self.output_path+file,'r') as t:
                
                lines=t.readlines()
                for line in lines:
                    
                    pres=float(line.split(' ')[0])
                    reca=float(line.split(' ')[1])
                    F=float(line.split(' ')[3])
                    pre.append(pres)
                    rec.append(reca)
                    FRP.append(F)
            
            print("file:{},pre_avg:{},rec_avg:{},FRP_avg:{},".format(file,mean(pre),mean(rec),mean(FRP))) 
    
# glass_confusion = ConfusionMatrix(num_classes=2, labels=['no_glass','glass'])
# pred_eye=[0,1,0,1,2,2]
# eye=[1,1,0,1,2,0]
# eye_confusion = ConfusionMatrix(num_classes=3, labels=['open','close','invisible'],preds=pred_eye, true_labels=eye)
# # eye_confusion.update(pred_eye, eye)

# eye_confusion.plot()
# eye_confusion.summary()
# class eval_index(object):


#     def __init__(self, num_classes: int, preds:list,labels: list):
#         self.matrix = np.zeros((num_classes, num_classes))#初始化混淆矩阵，元素都为0
#         self.num_classes = num_classes#类别数量，本例数据集类别为5
#         self.labels = labels#类别标签
#         self.preds=preds#预测结果
        
#     def roc_test(self):
# output_path='output/eval_result/'
# for file in os.listdir(output_path):
#     pre=[]
#     rec=[]
#     FRP=[]
#     with open (output_path+file,'r') as t:
        
#         lines=t.readlines()
#         for line in lines:
            
#             pres=float(line.split(' ')[1])
#             reca=float(line.split(' ')[2])
#             F=float(line.split(' ')[4])
#             pre.append(pres)
#             rec.append(reca)
#             FRP.append(F)
    
#     print("file:{},pre_avg:{},rec_avg:{},FRP_avg:{},".format(file,mean(pre),mean(rec),mean(FRP)))       
