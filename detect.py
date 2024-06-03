import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from detect_dataset import MyDataset
import torch
import shutil
import torch.nn as nn
import torchvision.utils
import yaml
import glob
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision.transforms import transforms
from torchvision import models
from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from torchsummary import summary
import torch.nn.functional as F
from PIL import Image
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass
import torchvision.models
from Triplet_model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
# device = torch.device('cpu')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
args = parser.parse_args()
import numpy as np
import math
save_path='/data/cifs/f/Dataset/eyes_state/taijiacaiji/video_frame_crop/20221220/detect/'

def padding(image):
    '''
    image: cv2 image, ndarray
    '''
    image=np.asarray(image)
    (height, width, channel) = image.shape
    long_side = height if height >= width else width
    new_image = np.zeros([long_side, long_side, channel], dtype=np.uint8)
    # new_image=np.full([long_side, long_side, channel],114,dtype=np.uint8)
    start_point_h = math.floor((long_side - height) / 2)
    start_point_w = math.floor((long_side - width) / 2)
    new_image[start_point_h:start_point_h+height, start_point_w:start_point_w+width, :] = image
    new_image = Image.fromarray(new_image)
    # cv2.imwrite('test1.jpg', new_image)    
    return new_image
# mbn_model=create_model('mobilenetv3_small_100')
# model=triplet_model2(mbn_model)
# mbn_model=models.mobilenet_v3_small()
# model=tri_mbnv3(mbn_model)
# mbn_model=models.mobilenet_v3_large()
# model=tri_mbnv3_large(mbn_model)
# model.load_state_dict(torch.load('./output/train/20221205-085825-mobilenetv3_large_100-128/checkpoint-16.pth.tar')['state_dict'])
model=torch.load('./output/train/20221220-093218-mobilenetv3_large_100-128/checkpoint-162.pth')
# load_checkpoint(model, './output/train/20221123-194125-mobilenetv3_small_100-128/checkpoint-135.pth.tar', use_ema=True)
# model=torch.load('output/train/20221121-155758-mobilenetv3_small_100-128/model_best.pth.tar')
model=model.to(device)
# load_checkpoint(model, 'output/train/20221110-172718-mobilenetv3_small_100-128/model_best.pth', use_ema=True)
transform = transforms.Compose([
                                    transforms.Resize((128, 128)),

                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
model.eval() 

glass_states=['no_glass','glass','sunglass']
eye_states=['open','close','invisible']

# img_list=glob.glob(r'img/*.bmp')
# img_path='/data/cifs/f/Dataset/eyes_state/taijiacaiji/video_frame_crop/20221220/1-5_sunglass-invisivle/'
img_path='/data/cifs/f/hzhhrd017a\F\Dataset\eyes_state\SV_test\val\dbht\glass\open'
img_list=os.listdir(img_path)
for img_name in img_list:
    print(img_name)
    img1 = Image.open(img_path+img_name).convert('RGB')
    img=padding(img1)
    image=transform(img)
    image=image.cuda()
    image= torch.unsqueeze(image, dim=0)
    print(image.shape)
    
    output = model(image)
    
    x_glass=output[0]
    x_eye=output[1]
    
    glass_probs = F.softmax(x_glass,dim=1).detach().cpu()
    eye_probs = F.softmax(x_eye,dim=1).detach().cpu()
    
    glass_score,pred_glass=torch.max(glass_probs,1)
    eye_score,pred_eye=torch.max(eye_probs,1)
    
    eye_pre = eye_states[int(pred_eye)]
    glass_pre=glass_states[int(pred_glass)]
    print({'glass_state': glass_pre,'eye_state': eye_pre,'eye_score':eye_score.item()})
    # if eye_pre != eyestate:
    if not  os.path.exists(os.path.join(save_path,glass_pre,eye_pre)) :
        os.makedirs(os.path.join(save_path,glass_pre,eye_pre))
    
    img1.save(os.path.join(save_path,glass_pre,eye_pre,img_name))
    
    # if eye_pre=='open' or eye_pre =='close':
        
    #     print({'eye_pre':eye_pre, 'glass_pre':glass_pre})
    #     img.save(os.path.join(else_path,eye_pre,img_name))
        
    # if glass_score.item() > 0.5 and eye_score.item() >0.5:
        
        
    #     img.save(os.path.join(save_path,glass_pre,eye_pre,img_name))
    # else:
    #     img.save(os.path.join(else_path,img_name)) 
        
        
        
        
        
        
        
        
        
        
        
'''
val:
# dataset = MyDataset(path,annotations_file,transform=transform)
# loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=True)
model.eval() 
glass_states=['no_glass','glass','sunglass']
eye_states=['open','close','invisible']
# for batch_idx, (input,img_path, target) in enumerate(loader):
#         print(img_path)
#         print(target)
#         # glass=target[:,0]
#         glass=target[0]
#         # print("glass",glass)
#         # print('glass.size',glass.size())
#         # eye=target[:,1]
#         eye=target[1]
#         glass=glass.cuda()
#         eye=eye.cuda()
#         input=input.cuda()
#         output = model(input)
#         x_glass=output[0]
#         # print('output',output)
#         x_eye=output[1]
#         _,pred_glass=torch.max(x_glass,1)
#         _,pred_eye=torch.max(x_eye,1)
#         if pred_glass==glass and pred_eye==eye :
#             print({'pred_glass':pred_glass,'pred_eye':pred_eye})
#         else :
#             img_path="".join(img_path)
#             # print()
#             # glass_name=glass_states.index(pred_glass.item())
#             # eye_name=eye_states.index(pred_eye.item())
#             eye_pre = eye_states[int(pred_eye)]
#             glass_pre=glass_states[int(pred_glass)]
#             eye_s = eye_states[int(eye)]
#             glass_s=glass_states[int(glass)]
#             shutil.copy(img_path,os.path.join(save_path,glass_s,eye_s,glass_pre+'_'+eye_pre+'_'+str(img_path).split('/')[-1]))
'''