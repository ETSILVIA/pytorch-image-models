import os
import matplotlib.pyplot as plt
from numpy import *
path='/workspace/pytorch-image-models/output/mobilenetv3_small_50_sigmoid_pre_recall.txt'
loss=[]
glass_top1=[]
eye_top1=[]
glass_precision=[]
eye_precision=[]
glass_recall=[]
eye_recall=[]
with open (path,'r') as f:
    lines=f.readlines() 
   
    
    for line in lines:
        
        line=line.split(' ')
        for i in line:
            if i==' ' or i=='':
                line.remove(i)
                # line.remove('')
        # print(line)
        if line[0]=='Test:' and line[1]!='(EMA):':
            # print(line)
            loss_item=float(line[7])
            glass_acc1=float(line[26])
            eye_acc1=float(line[29])
            glass_pre=float(line[10])
            eye_pre=float(line[18])
            glass_rec=float(line[14])
            eye_rec=float(line[22])
            glass_top1.append(glass_acc1)
            eye_top1.append(eye_acc1)
            # print(float(loss_item))
            
            loss.append(loss_item)
            glass_precision.append(glass_pre)
            glass_recall.append(glass_rec)
            eye_precision.append(eye_pre)
            eye_recall.append(eye_rec)

            # if loss_item.endswith('Loss:'):
            #     loss_item1=line[12]
            #     loss.append(loss_item1)
            # else:
glass_top1_m=mean(glass_top1)
eye_top1_m=mean(eye_top1)
print('glass_precision',mean(glass_precision))
print('eye_precision',mean(eye_precision))
print('glass_recall',mean(glass_recall))
print('eye_recall',mean(eye_recall))
print('eye_top1_m',eye_top1_m)
print('glass_top1_m',glass_top1_m)
iter = range(len(loss))           #     loss.append(loss_item)
fig = plt.figure(figsize=(20,26),dpi=1000)    #figsize是图片的大小`
ax1 = fig.add_subplot(3, 1, 1)
        
plt.plot(iter,loss,color='green',linewidth = 3)
ax1 = fig.add_subplot(3, 1, 2)

plt.plot(iter,glass_top1,color='green',linewidth = 3)
ax1 = fig.add_subplot(3, 1, 3)          
plt.plot(iter,eye_top1,color='green',linewidth = 3)

# plt.savefig('6.jpg')
            # test_loss=