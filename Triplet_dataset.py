

# from torch.utils.dataset import DataLoader

import numpy as np
import pandas as pd
from PIL import Image
import random
import math

 
def pad_image(image, target_size):
 
    """
    :param image: input image
    :param target_size: a tuple (num,num)
    :return: new image
    """
 
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
 
    # print("original size: ",(iw,ih))
    # print("new size: ", (w, h))
 
    scale = min(w / iw, h / ih)  # 转换的最小比例
 
    # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
    nw = int(iw * scale+0.5)
    nh = int(ih * scale+0.5)
 
    # print("now nums are: ", (nw, nh))
 
    image = image.resize((nw, nh))  # 更改图像尺寸，双立法插值效果很好
    #image.show()
    new_image = Image.new('RGB', target_size, (0,0,0))  # 生成白色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式
    #new_image.show()
 
    return new_image
def padding(image):
    '''
    image: cv2 image, ndarray
    '''
    image=np.asarray(image)
    (height, width, channel) = image.shape
    long_side = height if height >= width else width
    new_image = np.zeros([long_side, int(long_side), channel], dtype=np.uint8)
    # new_image=np.full([long_side, long_side, channel],114,dtype=np.uint8)
    start_point_h = math.floor((long_side - height) / 2)
    start_point_w = math.floor((long_side - width) / 2)
    new_image[start_point_h:start_point_h+height, start_point_w:start_point_w+width, :] = image
    new_image = Image.fromarray(new_image)
    # cv2.imwrite('test1.jpg', new_image)    
    return new_image
 
 
class TinyImageNetDataset_Triplet():
    def __init__(self, df,path, transform=None):
        self.data_csv = df
        self.transform = transform
        self.path = path       
        self.images = df.iloc[:, 0].values
        self.glass_labels = df.iloc[:, 1].values
        
        self.eye_labels = df.iloc[:, 2].values
      
        self.index = df.index.values 
       
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        eye_states=['open','close','invisible']
        glass_states=['no_glass','glass','sunglass']
        anchor_image_name = self.images[item]
        anchor_image_path = self.path + self.glass_labels[item] +'/' +self.eye_labels[item]+ '/' +anchor_image_name
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        anchor_glass_label = self.glass_labels[item]
        anchor_eye_label = self.eye_labels[item]
        # print(len(self.index[self.index!=item]))
        positive_list=list(set(self.index[self.index!=item][self.glass_labels[self.index!=item]==anchor_glass_label]).intersection(set(self.index[self.index!=item][self.eye_labels[self.index!=item]==anchor_eye_label])))
        positive_item = random.choice(positive_list)
        positive_image_name = self.images[positive_item]
        positive_image_path = self.path + self.glass_labels[positive_item] +'/' +self.eye_labels[positive_item] + '/' +positive_image_name
        positive_img = Image.open(positive_image_path).convert('RGB')
        negative_list=list(set(self.index[self.index!=item][self.glass_labels[self.index!=item]!=anchor_glass_label]).intersection(set(self.index[self.index!=item][self.eye_labels[self.index!=item]!=anchor_eye_label])))
        negative_item = random.choice(negative_list)
        negative_image_name = self.images[negative_item]
        negative_image_path = self.path  + self.glass_labels[negative_item] +'/' +self.eye_labels[negative_item] +'/' + negative_image_name
        negative_img = Image.open(negative_image_path).convert('RGB')
        #negative_img = self.images[negative_item].reshape(28, 28, 1)
        if self.transform!=None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)                   
            negative_img = self.transform(negative_img)
        label=[positive_img, negative_img,anchor_glass_label,anchor_eye_label]
        # print({'anchor':anchor_image_path,'pos':positive_image_path,'neg':negative_image_path})
        # print(label)
        # print([anchor_img, positive_img, negative_img,anchor_glass_label,anchor_eye_label])
        return (anchor_img, positive_img, negative_img),(int(glass_states.index(anchor_glass_label)),int(eye_states.index(anchor_eye_label)))
# train_data= pd.read_csv('label/val.csv')
# train_data_path = '/data/cifs/f/Dataset/eyes_state/new_data/'
# t=TinyImageNetDataset_Triplet(train_data,train_data_path)
# for i in range(50):
#     l=t.__getitem__(i)
#     print(l)

class glass_Dataset_Triplet():
    def __init__(self, df,path, transform=None):
        self.data_csv = df
        self.transform = transform
        self.path = path       
        self.images = df.iloc[:, 0].values
        self.glass_labels = df.iloc[:, 1].values
        
        self.eye_labels = df.iloc[:, 2].values
      
        self.index = df.index.values 
       
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        eye_states=['01_Open','00_Close','02_Invisible']
        glass_states=['no_glass','glass','sunglass']
        anchor_image_name = self.images[item]
        anchor_image_path = self.path  +anchor_image_name
        # print(anchor_image_path)
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        # print(anchor_img)
        anchor_img=pad_image(anchor_img,(128,128))
        # anchor_img=padding(anchor_img)
        # print(anchor_img)
        # print(anchor_img)
        anchor_img.save('pad1.jpg')
        anchor_glass_label = self.glass_labels[item]
        # print(anchor_glass_label)
        anchor_eye_label = self.eye_labels[item]
        # print(anchor_eye_label)
        # print('*******************************8',anchor_eye_label)
        # if anchor_eye_label in ['myopia_glass_squint','myopia_noglass_squint','squint','single_eye']:
        #     # print('*******************************8',anchor_eye_label)
        #     anchor_eye_label='open'
        # if anchor_eye_label in ['myopia_glass_close','myopia_noglass_close']:
        #     anchor_eye_label='close'
        # elif anchor_eye_label=='invisible1':
        #     anchor_eye_label='invisible'

        if self.transform!=None:
            anchor_img = self.transform(anchor_img)
         
        return anchor_img,(int(glass_states.index(anchor_glass_label)),int(eye_states.index(anchor_eye_label)))
    
# train_data= pd.read_csv('label/train_20231005.csv')

# train_data_path = '/data/cifs/f/Dataset/eyes_state/003_new_data/train_2022-12-13/'
# t=glass_Dataset_Triplet(train_data,train_data_path)
# for i in range(80000):
#     l=t.__getitem__(i)
#     print(l)