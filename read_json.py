# 读取标注数据中的json标签，将bbox转成yolo的标签格式txt
# 删除掉没有标注的图片，将图片重新保存至新文件夹中

import os
import random
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join
import shutil
import json
import numpy as np
import cv2

def convert(size, box):
    dw = 1. / float(size[0])
    dh = 1. / float(size[1])
    x_center = box[0]+box[2]/2
    y_center = box[1]+box[3]/2
    w = box[2]
    h = box[3]
    x = x_center * dw
    w = w * dw
    y = y_center * dh
    h = h * dh
    return (x, y, w, h)
  
def convert_annotation(image_id):
    # in_file = json.load(open(rootpath+'/%s' % (image_id), "r", encoding="utf-8"))
    # out_file = open(savepath+'/%s.txt' % (image_id[:-5]), 'w')
    
    # img_path = os.path.join(imgpath+'/%s.jpg' % (image_id))
    # json_filename = os.path.join(rootpath+'/%s.json' % (image_id))

    json_filename=image_id
    json_file = json.load(open(json_filename+'.json', "r", encoding="utf-8"))
    save_name=image_id.split("/")[-1]
    # print(json_file['shapes'])
    print(image_id)
    # img = cv2.imread(img_path)
    # w, h = img.shape[1], img.shape[0]
    w = json_file['imageWidth']
    h = json_file['imageHeight']
    # 剔除空白标注文件
    
    all_cls=[]
    for obj in json_file['shapes']:
        cls = obj['label'].lower()
        if cls=='object':
            cls=obj['object2']
        all_cls.append(cls)
    if 'phone' in all_cls:
    # if all_cls is not None:
        if json_file['shapes'] != []:
            out_file = open(savepath+'/%s.txt' % (save_name.replace('.','')), 'w')
        # copy_pics(image_id)
        for obj in json_file['shapes']:
            cls = obj['label'].lower()
            if cls=='object':
                cls=obj['object2']
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            bbox = obj['points']
            # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
            #      float(xmlbox.find('ymax').text))
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            if len(bbox) == 0:
                continue
            bb = convert((w, h), bbox)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
# def copy_pics(image_id):
#     img_filename = os.path.join(imgpath + '/%s.jpg' % image_id)
#     out_path = os.path.join(save_imgpath + '/%s.jpg' % image_id)
#     shutil.copy(img_filename, out_path)
    
# for x in range(1, 2):   
if __name__=='__main__': 
    import glob
    rootpath = r'/data/cifs/d/Camera/DM0X/15_DataProviderDeliveredSamples/SpeechOcean_DMS_Labeled/Test_data/语音_7523/'
    savepath = r'/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/dbht/speechocean_dms_labeled/object_detection/test/wechatting/txt/'
    # or_path=r'/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/dbht/speechocean_dms_labeled/object_detection/normal/txt/'
    # exist_list=os.listdir(or_path)
    # imgpath = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_jundong/美妆镜数据待标注_20230105_new/1/images'新建文件夹.7z
    # save_imgpath = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_jundong/datasets/labels/8_images'
    # classes = ['Trunk', 'Face', 'Hand', 'lipstick', 'powder', 'eyebrow', 'comb', 'lip_gloss', 'eyebrow_brush', 'powder_box'] 
    classes = ['trunk', 'face', 'hand', 'milk', 'bottle', 'cup', 'cigarette', 'phone']
    # classes = ['eye']    
    # files = os.listdir(rootpath)
    files=glob.glob(rootpath+'/*.json')
    imgs=glob.glob(rootpath+'/*.jpg')
    img_list=os.listdir(savepath)
    for json_file_ in files:
        id=json_file_.split('/')[-2]
        
        json_name=json_file_.split('/')[-1]
        txt_name=json_name.replace('.json','.txt')
        # if txt_name not in exist_list:
        # if id in['1','2','3','4']:
        if json_file_.replace('.json','.jpg') in imgs:
            convert_annotation(json_file_.split('.json')[0])
    # import os
    # path=r'\\hzhhrd017a\F\Dataset\2022_8_23_data_label\wang_hongbo\20230220\process\dbht\ITT_DMS_20230601\train\Labeled_data\txt_new'
    # img_names=os.listdir(path)
    # for img_name in img_names:
    #     rename=img_name[:-4]
    #     rename=rename.replace('.','')
    #     print(rename)
    #     os.rename(os.path.join(path,img_name),os.path.join(path,rename+'.txt'))
    # json_filename = os.path.join(rootpath, json_file_)
    # json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    # # print(json_file['info'])
    # for multi in json_file["shapes"]:
    #         points = np.array(multi["points"])
    #         labelName=multi["label"]
    #         x1 = points[0]
    #         y1 = points[1]
    #         x2 = points[2] +  points[0]
    #         y2 = points[3] + points[1]
    #         print(x1 ,y1 ,x2 ,y2)
    #         start = (int(x1), int(y1))
    #         end = (int(x2), int(y2))
    #         color = (255, 0, 0) 
    #         thickness = 2
    #         img = cv2.imread(os.path.join(imgpath, json_file_.replace('.json', '.jpg')))
    #         image = cv2.rectangle(img, start, end, color, thickness) 
    #         cv2.imwrite('/workspace/wjd/yolov5-master/tools/1.jpg', image)
            

# 对DMS手势标注数据集的json标签进行转换
# def convert(size, box):
#     dw = 1. / size[0]
#     dh = 1. / size[1]
#     x_center = (box[0] + box[2]) / 2
#     y_center = (box[1] + box[3]) / 2
#     w = box[2] - box[0]
#     h = box[3] - box[1]
#     x = x_center * dw
#     w = w * dw
#     y = y_center * dh
#     h = h * dh
#     return (x, y, w, h)
  
# def convert_annotation(image_id):
#     # in_file = json.load(open(rootpath+'/%s' % (image_id), "r", encoding="utf-8"))
#     # out_file = open(savepath+'/%s.txt' % (image_id[:-5]), 'w')
    
#     json_filename = os.path.join(rootpath+'/%s.json' % (image_id))
#     json_file = json.load(open(json_filename, "r", encoding="utf-8"))
#     img_path = os.path.join(imgpath+'/%s.jpg' % (image_id))
#     # print(json_file['shapes'])
#     print(image_id)
    
#     img = cv2.imread(img_path)
#     w = img.shape[1]
#     h = img.shape[0]
#     # 剔除空白标注文件
#     if json_file['info'] != []:
#         out_file = open(savepath+'/%s.txt' % (image_id), 'w')
#         # copy_pics(image_id)
#     for multi in json_file["info"]:
#         try:
#             for hand in multi["hands"]:
#                 cls_id = "2"
#                 bbox = hand["hand_bbox"]
#                 # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#                 #      float(xmlbox.find('ymax').text))
#                 if len(bbox) == 0:
#                     continue
#                 bb = convert((w, h), bbox)
#                 out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
#         except:
#             pass
#         try:
#             for face in multi["faces"]:
#                 cls_id = "1"
#                 bbox = face["face_bbox"]
#                 # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#                 #      float(xmlbox.find('ymax').text))
#                 if len(bbox) == 0:
#                     continue
#                 bb = convert((w, h), bbox)
#                 out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
#         except:
#             pass
        
# # def copy_pics(image_id):
# #     img_filename = os.path.join(imgpath + '/%s.jpg' % image_id)
# #     out_path = os.path.join(save_imgpath + '/%s.jpg' % image_id)
# #     shutil.copy(img_filename, out_path)
    
    
# rootpath = '/data/cifs/f/Dataset/detection_dataset_wjd/DMS_手势数据/json/'
# savepath = '/data/cifs/f/Dataset/detection_dataset_wjd/DMS_手势数据/label'
# imgpath = '/data/cifs/f/Dataset/detection_dataset_wjd/DMS_手势数据/images'
# # save_imgpath = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_jundong/datasets/labels/4_images'
# classes = ['Trunk', 'faces', 'hands']       
# files = os.listdir(rootpath)
# for json_file_ in files:
#     convert_annotation(json_file_.split('.')[0])

    # json_filename = rootpath + json_file_
    # json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    # # print(json_file['info'])
    # for multi in json_file["info"]:
    #     for hand in multi["hands"]:
    #         points = np.array(hand["hand_bbox"])
    #         labelName=["hand_bbox"]
    #         x1 = points[0]
    #         y1 = points[1]
    #         x2 = points[2]
    #         y2 = points[3]
    #         print(x1 ,y1 ,x2 ,y2)
    #         start = (int(x1), int(y1))
    #         end = (int(x2), int(y2))
    #         color = (255, 0, 0) 
    #         thickness = 2
    #         img = cv2.imread('/data/cifs/f/Dataset/detection_dataset_wjd/DMS_手势数据/images/20200724132739_20200724132740_1595568460_35.jpg')
    #         image = cv2.rectangle(img, start, end, color, thickness) 
    #         cv2.imwrite('/workspace/wjd/yolov5-master/tools/1.jpg', image)
            