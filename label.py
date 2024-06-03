import csv
import os
import tqdm
import glob
data_path='/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/train'
# data_path='/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/quantization_100000/'
glass_state_label=os.listdir(data_path)
eye_states=['01_Open','00_Close','02_Invisible']
# eye_states=['open','close','invisible','squint']
glass_states=['no_glass','glass','sunglass']
img_list=glob.glob(data_path+'/*/*/*/*.png')
label={"01_Open":0,"00_Close":0,"02_Invisible":0,"no_glass":0,"glass":0,"sunglass":0}
for img in img_list:
     glass_label=img.split('/')[-4]
     eye_label=img.split('/')[-3]
     label[glass_label] += 1
     label[eye_label] += 1
     img_path=img.replace(data_path,'')
     # if  ' ' not in img_path:
     with open ("label/single_eye_train_20240105.csv","a+") as f:
               writer=csv.writer(f)
               writer.writerow([img_path,glass_label,eye_label])
    #  print(eye_label)
print(label)

            
                           
# print(img_len)