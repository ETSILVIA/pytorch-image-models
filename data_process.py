# from glob import glob     
# path1='/data/cifs/f/Dataset/eyes_state/003_new_data/glass'
# path2='/data/cifs/f/Dataset/eyes_state/003_new_data/val_2022-11-21'
# train_imgs= glob(path1+'/*/*/*.jpg')
# val_img= glob(path2+'/*/*/*.jpg')

# train_list={}
# train_name=[]

# for train_img in train_imgs:
#     # print(train_img)
#     train_img_name=train_img.split('/')[-1]
#     print(train_img_name)
#     train_name.append(train_img_name)
#     # print(train_img)
#     train_list[train_img_name]=train_img
# idx=0
# print(len(val_img))
# for img_path in val_img:
    
#     val_img_name=img_path.split('/')[-1]

#     print(val_img_name)
#     if val_img_name in train_img:
#         idx+=1
#         print(val_img_name)
# print(idx)
import csv
import pandas as pd
train_csv='label/train.csv'
val_csv='label/val.csv'
train_data=pd.read_csv(train_csv)
val_data=pd.read_csv(val_csv)
train_images = train_data.iloc[:, 0].values.tolist()
# print(train_images.tolist())
val_images = val_data.iloc[:, 0].values.tolist()
idx=0
for val_image in val_images:
    if val_image in train_images:
        try:
            train_data=train_data.drop([train_images.index(val_image)])
            idx+=1
        except:
            pass
train_data.to_csv('label/drop_train.csv',index=False,encoding="utf-8")
print(idx)