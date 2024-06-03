import torch.nn as nn
import torch 
from timm.models import create_model
from torchvision import models
# import efficientnet_pytorch
# model = create_model(
        # model_name='mobilenetv3_small_100')
# model.glass=nn.Linear(1024,512,bias=True)
# model.glass_cls=nn.Linear(512,3,bias=True)

# model.eye=nn.Linear(1024,512,bias=True)
# model.eye_cls=nn.Linear(512,3)
# mm=model
# print(mm)
class triplet_model(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # num_filters = self.Feature_Extractor.glass
            self.Triplet_Loss_glass=nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(),nn.Linear(512,10),nn.Linear(10,2))
            self.Triplet_Loss_eye=nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(),nn.Linear(512,10),nn.Linear(10,2))
        def forward(self,x):
            x1=self.Feature_Extractor(x)
            # print(x.size())
            triplets_glass = self.Triplet_Loss_glass(x1)
            triplets_eye = self.Triplet_Loss_glass(x1)
            return triplets_glass,triplets_eye
# mm=triplet_model(model)   
# print(mm)    
# timm框架中mbnv3_small_100
class triplet_model2(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # num_filters = self.Feature_Extractor.glas
            # print(self.Feature_Extractor)
            self.triple_eye=nn.Linear(1024,256)
            self.triple_glass=nn.Linear(1024,256)
            self.glass=nn.Sequential(nn.Linear(256,3))
            self.eye=nn.Sequential(nn.Linear(256,3))
        def forward(self,x):
            x1=self.Feature_Extractor(x)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            glass=self.glass(x2)
            eye=self.eye(x3)
            return glass,eye,x2,x3
# timm框架中mbnv3_large_100
class triplet_model2_large(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # num_filters = self.Feature_Extractor.glas
            # print(self.Feature_Extractor)
            self.triple_eye=nn.Linear(1280,256)
            self.triple_glass=nn.Linear(1280,256)
            self.glass=nn.Sequential(nn.Linear(256,3))
            self.eye=nn.Sequential(nn.Linear(256,3))
        def forward(self,x):
            x1=self.Feature_Extractor(x)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            glass=self.glass(x2)
            eye=self.eye(x3)
            return glass,eye,x2,x3
class timm_model(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # num_filters = self.Feature_Extractor.glas
            # print(self.Feature_Extractor)
            # self.triple_eye=nn.Linear(1024,256)
            # self.triple_glass=nn.Linear(1024,256)
            self.glass=nn.Sequential(nn.Linear(1024,3))
            self.eye=nn.Sequential(nn.Linear(1024,3))
        def forward(self,x):
            x1=self.Feature_Extractor(x)
            
            glass=self.glass(x1)
            eye=self.eye(x1)
            return glass,eye        
class triplet_model3(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # num_filters = self.Feature_Extractor.glas
            # print(self.Feature_Extractor)
            self.triple_eye=nn.Sequential(nn.Linear(1024,2))
            self.triple_glass=nn.Sequential(nn.Linear(1024,2))
            self.glass=nn.Sequential(nn.Linear(2,3))
            self.eye=nn.Sequential(nn.Linear(2,3))
        def forward(self,x):
            x1=self.Feature_Extractor(x)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            glass=self.glass(x2)
            eye=self.eye(x3)
            return glass,eye,x3
# mm=triplet_model2(model)
# print(mm)        
 #在torch 官方封装的mobilenetv3基础上将最后一层classfier 变为 多分支分类  
from torchvision import models
import torch.nn as nn
import torch
class mbn(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            self.Triplet_Loss_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            self.Triplet_Loss_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x=self.Feature_Extractor(x)
            x1 = torch.flatten(x, 1)
            triplets_glass = self.Triplet_Loss_glass(x1)
            triplets_eye = self.Triplet_Loss_glass(x1)
            return triplets_glass,triplets_eye

# model=models.mobilenet_v3_small(pretrained=True)

class tri_mbnv3(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            self.conv_head=nn.Conv2d(576,1024,kernel_size=(1,1),stride=(1,1))
            self.act2=nn.Hardswish()
            self.triple_eye=nn.Linear(1024,256)
            self.triple_glass=nn.Linear(1024,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x = self.Feature_Extractor(x)
            x = self.conv_head(x)
            x = self.act2(x)
            x1 = torch.flatten(x, 1)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            triplets_glass = self.glass(x2)
            triplets_eye = self.eye(x3)
            return triplets_glass,triplets_eye,x2,x3
        
class tri_mbnv3_large(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            self.conv_head=nn.Conv2d(960,1280,kernel_size=(1,1),stride=(1,1))
            self.act2=nn.Hardswish()
            self.triple_eye=nn.Linear(1280,256)
            self.triple_glass=nn.Linear(1280,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x = self.Feature_Extractor(x)
            x = self.conv_head(x)
            x = self.act2(x)
            x1 = torch.flatten(x, 1)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            triplets_glass = self.glass(x2)
            triplets_eye = self.eye(x3)
            return triplets_glass,triplets_eye,x2,x3       
class efficientnet(nn.Module):
    def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            self.conv_head=nn.Conv2d(576,1024,kernel_size=(1,1),stride=(1,1))
            self.act2=nn.Hardswish()
            self.triple_eye=nn.Linear(1024,256)
            self.triple_glass=nn.Linear(1024,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
    def forward(self,x):
        x = self.Feature_Extractor(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x1 = torch.flatten(x, 1)
        x2=self.triple_glass(x1)
        x3=self.triple_eye(x1)
        triplets_glass = self.glass(x2)
        triplets_eye = self.eye(x3)
        return triplets_glass,triplets_eye,x2,x3
class shuffenet(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            # print(self.Feature_Extractor)
            # self.triple_eye=nn.Linear(2048,256) inception
            # self.triple_glass=nn.Linear(2048,256)
            # self.triple_eye=nn.Linear(1280,256) efficienet
            # self.triple_glass=nn.Linear(1280,256)
            self.triple_eye=nn.Linear(1024,256)
            self.triple_glass=nn.Linear(1024,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x = self.Feature_Extractor(x)
            x1 = x.mean([2, 3])
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            triplets_glass = self.glass(x2)
            triplets_eye = self.eye(x3)
            return triplets_glass,triplets_eye,x2,x3
class incept(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-1])
            print(self.Feature_Extractor)
            self.triple_eye=nn.Linear(2048,256)
            self.triple_glass=nn.Linear(2048,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x = self.Feature_Extractor(x)
            x1 = torch.flatten(x, 1)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            triplets_glass = self.glass(x2)
            triplets_eye = self.eye(x3)
            return triplets_glass,triplets_eye,x2,x3
'''
import torch
import torch
from efficientnet_pytorch.model import EfficientNet
inputs = torch.rand(1, 3, 224, 224)
model = EfficientNet.from_pretrained('efficientnet-b0')
# model.eval()
# outputs = model(inputs)
# print(model)
class effnet(nn.Module):
    def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-2])
            # print(self.Feature_Extractor)
            self.triple_eye=nn.Linear(1280,256)
            self.triple_glass=nn.Linear(1280,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            self._swish = model._swish
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
    def forward(self,x):
        x = self.Feature_Extractor(x)
        print(x)
        x1 = torch.flatten(x, 1)
        x2=self.triple_glass(x1)
        x3=self.triple_eye(x1)
        triplets_glass = self.glass(x2)
        triplets_eye = self.eye(x3)
        return triplets_glass,triplets_eye,x2,x3
model1=effnet(model)
# print(model1)
inputs = torch.rand(1, 3, 224, 224)
model1.eval()
outputs = model1(inputs)
print(outputs)
'''
# model.classfier=(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
# print(model.classfier)
# print(model)
# mm=mbn(model)
# mm.forward(torch.rand(16,3,3,3))   
# print(mm)  
import torch
import torch.nn as nn
import torch 
from torchvision import models
from timm.models import create_model
class incept1(nn.Module):
        def __init__(self,model):
            super().__init__()
            # num_f=model.fc.in_features
            self.Feature_Extractor=nn.Sequential(*list(model.children())[:-5])
            print(self.Feature_Extractor)
            self.triple_eye=nn.Linear(768,256)
            self.triple_glass=nn.Linear(768,256)
            self.glass=nn.Linear(256,3)
            self.eye=nn.Linear(256,3)
            # self.Triplet_glass=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
            # self.Triplet_eye=nn.Sequential(nn.Linear(576,1024),nn.Hardswish(),nn.Dropout(p=0.2, inplace=True),nn.Linear(1024,3))
        def forward(self,x):
            x = self.Feature_Extractor(x)
            print(x.size())
            x1 = torch.flatten(x, 1)
            x2=self.triple_glass(x1)
            x3=self.triple_eye(x1)
            triplets_glass = self.glass(x2)
            triplets_eye = self.eye(x3)
            return triplets_glass,triplets_eye,x2,x3
        
# model1=models.inception_v3(pretrained=True)
# # model=incept1(model1)
# print(model1)
# inputs = torch.rand(8, 3, 299, 299)
# # model.eval()
# outputs = model1(inputs)
# print(outputs)