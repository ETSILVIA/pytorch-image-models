import pandas as pd
import random
import csv
import cv2
import os
train_data= pd.read_csv('label/train_20231128.csv')
train_data_path = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/train/'
save_path='/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/quantization_100000/'
# print(type(train_data))
with open('/workspace/eye_net/single_eye_state/label/single_eye_train_20231218.csv', 'r') as file:
    reader = csv.reader(file)
    rows = [row for row in reader]
sample_rows = random.sample(rows, 100000)  # 从所有行中随机采样2行
for row in sample_rows:
    # print(row)
    img_path=row[0]
    glass_state=row[1]
    eye_state=row[2]
    img=cv2.imread(train_data_path+img_path)
    path_split=img_path.split('/')
    save_str=path_split[1]+'/'+path_split[2]+'/'+path_split[3]+'/'
    if not os.path.exists(save_path+save_str):
        os.makedirs(save_path+save_str)
  
    print(save_path+img_path)
    cv2.imwrite(save_path+img_path.replace(' ','_'),img)
    


# choice_num=random.choice(train_data)
# print(choice_num)
# sample = train_data.sample(frac=0.1, random_state=5, axis=0)
# print(sample)
# for i in range(1000):
#     random_index=random.randint(0,360000)
#     print(random_index)
#     print(train_data[random_index])

# random.shuffle(csv_reader)

