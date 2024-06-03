import pandas as pd
import random
import csv
import shutil
import cv2
import os
# train_data= pd.read_csv('//V01/dfs/DIDA6003/CT/CT Projects/Platform/G9PH Platform/07_DEQ/57_others/Trinity问题点log/ZhangHongyang/IMS02第一轮测试集/EyeNet/EyeNet_all/filelist.txt')
# f=open('//V01/dfs/DIDA6003/CT/CT Projects/Platform/G9PH Platform/07_DEQ/57_others/Trinity问题点log/ZhangHongyang/IMS02第一轮测试集/EyeNet/EyeNet_all/filelist.txt')
# train_data=f.readlines()
# train_data_path = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/train/'
# train_data_path='//V01/dfs/DIDA6003/CT/CT Projects/Platform/G9PH Platform/07_DEQ/57_others/Trinity问题点log/ZhangHongyang/IMS02第一轮测试集/EyeNet/EyeNet_all/'
# save_path='/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/quantization_80000/'
# save_path='//V01/dfs/DIDA6003/CT/CT Projects/Platform/G9PH Platform/07_DEQ/57_others/Trinity问题点log/ZhangHongyang/IMS02第一轮测试集/EyeNet/EyeNet_3w/'
# print(type(train_data))
train_data_path='/data/cifs/f/Dataset/eyes_state/00_Singleye_Dataset_0513/train/'
save_path='/data/cifs/f/Dataset/eyes_state/qua_8255/'
rows=[]
with open('/workspace/eye_net/single_eye_state/label/single_eye_train_20240513_new.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print (row)
        if '8255' in row[0]:
            rows.append(row)
    # print(rows)
    # rows = [row for row in reader]
sample_rows = random.sample(rows, 800) 
for row in sample_rows:
    # print(row)
    img_path=row[0]
    # glass_state=row[1]
    # eye_state=row[2]
    # print(train_data_path+img_path)
    
    path_split=img_path.split('/')
    print(path_split)
    image_name=path_split[-1]
    save_str=path_split[1]+'/'+path_split[2]+'/'+path_split[3]+'/'
    
    # img=cv2.imread(train_data_path+save_str+image_name)
    if not os.path.exists(save_path+save_str):
        os.makedirs(save_path+save_str)
  
    print(save_path+img_path)
    shutil.copy(train_data_path+save_str+image_name,save_path+save_str+image_name)
    # cv2.imwrite(save_path+save_str+image_name,img)
    


# choice_num=random.choice(train_data)
# print(choice_num)
# sample = train_data.sample(frac=0.1, random_state=5, axis=0)
# print(sample)
# for i in range(1000):
#     random_index=random.randint(0,360000)
#     print(random_index)
#     print(train_data[random_index])

# random.shuffle(csv_reader)

