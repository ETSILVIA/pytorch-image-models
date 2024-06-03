import mediapipe as mp
import cv2
import numpy as np
import os
import math
from mediapipe.python.solutions import face_mesh_connections
from itertools import chain
import argparse
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision.transforms import transforms
from timm.models import create_model, load_checkpoint
import torch.nn.functional as F
from PIL import Image
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--checkpoint', default='/workspace/pytorch-image-models/output/train/20220929-140453-mobilenetv3_small_100-128/last.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

#eyes_connections 坐标在face_mesh_connections.py中
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

# img_path = '/data/cifs/f/Dataset/eyes_state/denger_lizhuo/error_9_27/else'

# crop_path='/data/cifs/f/Dataset/eyes_state/denger_lizhuo/error_9_27/recrop_else/'
# if not os.path.exists(crop_path):
#         os.makedirs(crop_path)

LEFT_EYE =list(chain.from_iterable(face_mesh_connections.FACEMESH_LEFT_EYE))
LEFT_IRIS = list(chain.from_iterable(face_mesh_connections.FACEMESH_LEFT_IRIS))
LEFT_EYEBROW=list(chain.from_iterable(face_mesh_connections.FACEMESH_LEFT_EYEBROW))
RIGHT_EYE = list(chain.from_iterable(face_mesh_connections.FACEMESH_RIGHT_EYE))
RIGHT_IRIS = list(chain.from_iterable(face_mesh_connections.FACEMESH_RIGHT_IRIS))
RIGHT_EYEBROW=list(chain.from_iterable(face_mesh_connections.FACEMESH_RIGHT_EYEBROW))
face=list(chain.from_iterable(face_mesh_connections.FACEMESH_FACE_OVAL))

def class_result(args,crop_img,img_name):
    model=create_model(model_name='mobilenetv3_small_100')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
        

    load_checkpoint(model, args.checkpoint, use_ema=True)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    
                                        
                                    transforms.ToTensor(),
                                        ])

    model.eval() 
    glass_states=['no_glass','glass','sunglass']
    eye_states=['open','close','invisible']
    image=transform(crop_img)
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
    # print({'glass_score':glass_score.item(),'eye_score':eye_score.item()})
    if glass_pre != 'sunglass' or eye_pre != 'invisible':
        if glass_score.item() > 0.5 and eye_score.item() >0.5:
            cls_result=eye_pre+' '+str(eye_score.item())+' '+glass_pre+' '+str(glass_score.item())
            
            # cv2.imwrite(os.path.join(save_path,glass_pre,eye_pre,img_name),crop_img)
        else:
            cls_result=' '
        # cv2.imwrite(os.path.join(else_path,img_name),crop_img)
        return cls_result  
        
def find_index(list):
    iris=np.array(list)
    left_point_x = np.min(iris[:, 0])
    left_point_y = iris[:, 1][np.where(iris[:, 0] == left_point_x)][0] 
    right_point_x = np.max(iris[:, 0])
    right_point_y = iris[:, 1][np.where(iris[:, 0] == right_point_x)][0]
    instance=math.sqrt((right_point_x - left_point_x)**2 + (right_point_y - left_point_y)**2)
    return instance

def eye_state(iris_list):
    '''
    left_point_x = np.min(iris[:, 0])
    right_point_x = np.max(iris[:, 0])
    top_point_y = np.min(iris[:, 1])
    bottom_point_y = np.max(iris[:, 1])
    
    left_point_y = iris[:, 1][np.where(iris[:, 0] == left_point_x)][0]
    right_point_y = iris[:, 1][np.where(iris[:, 0] == right_point_x)][0]
    top_point_x = iris[:, 0][np.where(iris[:, 1] == top_point_y)][0]
    bottom_point_x = iris[:, 0][np.where(iris[:, 1] == bottom_point_y)][0]
    x_instance=math.sqrt((right_point_x - left_point_x)**2 + (right_point_y - left_point_y)**2)
    y_instance=math.sqrt((right_point_x - left_point_x)**2 + (right_point_y - left_point_y)**2)
    '''
    y_point=[]
    iris=np.array(iris_list)
    left_point_x = np.min(iris[:, 0])
    print('***',left_point_x)
    left_point_y = iris[:, 1][np.where(iris[:, 0] == left_point_x)][0] 
    right_point_x = np.max(iris[:, 0])
    right_point_y = iris[:, 1][np.where(iris[:, 0] == right_point_x)][0]
    x_instance=math.sqrt((right_point_x - left_point_x)**2 + (right_point_y - left_point_y)**2) 
    # for x in iris_list:
    #     if not operator.eq(x,(left_point_x,left_point_y) ) and not operator.eq(x,(right_point_x,right_point_y) ) :
    #        y_point.append(x)
    #        print('*!!*',y_point)
    # y_instance= find_index(np.array(y_point))
    # print('**',y_instance)
    # thre=float(y_instance)/float(x_instance )
    # if thre >1 :
    #     thre=float(1/thre)
    # print('*****',thre)
    
    # if thre < 0.15 or thre >1:
    #     eye_stat='close'
    # elif thre > 0.6  :
    #     eye_stat='open'
    # else:
    #     eye_stat='inv'
    # print(eye_stat)
    return left_point_x,left_point_y,right_point_x,right_point_y

def save_img(img,eye_state,path,img_name):

    cv2.imwrite(os.path.join(path,eye_state,img_name),img)
       
def imagecrop(image, box,img_name,angle,eye_center):
    print('angle',angle)
    # 获取四个顶点坐标

    angle = math.radians(angle)
    # print('box',box)
    x_center,y_center=eye_center
    xs=[]
    ys=[]
    for co in box:
        xo=co[0]
        yo=co[1]
        
        # x = int(x_center + math.cos(angle) * (xo - x_center) - math.sin(angle) * (yo - y_center))
        # y = int(y_center + math.sin(angle) * (xo - x_center) + math.cos(angle) * (yo - y_center))
        x = round(x_center + math.cos(angle) * (xo - x_center) + math.sin(angle) * (yo - y_center))
        y = round(y_center + math.sin(angle) * (xo - x_center) - math.cos(angle) * (yo - y_center))
        xs.append(x)
        ys.append(y)

    print(min(xs), max(xs), min(ys), max(ys))
    cropimage = image[min(ys):max(ys),min(xs):max(xs)]
    
    # print('======',cropimage)
    # cv2.drawContours(image, [box], 0, (255, 0, 0), 1)
    # cv2.imwrite(os.path.join(crop_path, img_name), cropimage)
    # try:
    #     # print('path',os.path.join(crop_path,img_name))
    #     # img=cv2.resize((256,256),cropimage)
    #     cv2.imwrite(os.path.join(crop_path,img_name), cropimage)

    # except:
    #     print('img_error', img_name)
    #     pass
    return cropimage

def crop_rect(img, rect):
  center, size, angle = rect[0], rect[1], rect[2]
  if(-45<angle<0 or angle>45):
      center=tuple(map(int,center))
      size = tuple([int(rect[1][1]), int(rect[1][0])])
      angle -= 90
      height, width = img.shape[0], img.shape[1]
      M = cv2.getRotationMatrix2D(center, angle, 1)
      img_rot = cv2.warpAffine(img, M, (width, height))
      # cv2.imwrite("1.jpg", img_rot)
      img_crop = cv2.getRectSubPix(img_rot, size, center)
  else:
      center, size = tuple(map(int, center)), tuple(map(int, size))
      # angle+=90
      height, width = img.shape[0], img.shape[1]
      M = cv2.getRotationMatrix2D(center, angle, 1)
    # size = tuple([int(rect[1][1]), int(rect[1][0])])
      img_rot = cv2.warpAffine(img, M, (width, height))
      # cv2.imwrite("2.jpg", img_rot)
      img_crop = cv2.getRectSubPix(img_rot, size, center)
      
  return img_crop, img_rot

def align_face(image_array,left_iri,right_iri):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    # calculate the mean point of landmarks of left and right eye
    print(left_iri)
    left_eye_center = np.mean(left_iri, axis=0).astype("int")
    print(left_eye_center)
    right_eye_center = np.mean(right_iri, axis=0).astype("int")
    print(right_eye_center)
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    print(left_eye_center[1] + right_eye_center[1])
    eye_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                  int(left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    # print('eye_center',eye_center)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def main():
    right_eyes=[]
    left_eyes=[]
    face_list=[]
    # img_names,img_list=get_img_list(img_path)
    # img_list=glob.glob(r'/data/cifs/f/Dataset/eyes_state/image/*/*/*.jpg')
    # select_list=os.listdir(img_path)
    # for idx, img_name in enumerate(img_list):
    
    cap=cv2.VideoCapture('/data/cifs/f/Dataset/DangerBehavior/video_ir/WIN_20220930_15_50_10_Pro.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc      = cv2.VideoWriter_fourcc(*'XVID')
    videowriter      = cv2.VideoWriter('output/result/WIN_20220930_15_50_10_Pro.mp4' ,fourcc, frame_count, (int(frame_width),int(frame_height)), True)
    
    count=0
    while(True):
        ret, frame = cap.read()
        if ret:
            # if(count % f_count == 0):
            #     print("开始截取视频第：" + str(count) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                # cv2.imwrite("./frame_folder/" + str(count) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            count += 1
            # cv2.waitKey(0)
	
            # image_name=img_name.split('/')[-1]
            image_name=str(count) + '.jpg'
            # if int(image_name.split('_')[0])>=18 and int(image_name.split('_')[0])<=70:
            # print(image_name)
            # print(img_name)
            # if image_name in select_list:
            #     img = cv2.imread(img_name)
            if frame is not None:
                h2, w2, c2 = frame.shape

                with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as face_mesh:
                    results = face_mesh.process(frame)
                    eyes = []
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            left_iri = []
                            right_iri = []
                            for idx, landmark in enumerate(face_landmarks.landmark):
                                if idx in face:
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    face_list.append((x1, y1))
                                if idx in LEFT_IRIS:  # 左虹膜
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    left_iri.append((x1, y1))

                                if idx in RIGHT_IRIS:  # 右虹膜
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    right_iri.append((x1, y1))
                                if idx in LEFT_EYEBROW or idx in RIGHT_EYEBROW:  # 中间区域
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    eyes.append((x1, y1))
                                if idx in [9,5,21,251,68,298,330,101] :  # 中间区域
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    eyes.append((x1, y1))
                                if idx in LEFT_EYE :  # 左眼,左眉
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    left_eyes.append((x1, y1))
                                    eyes.append((x1, y1))
                                if idx in RIGHT_EYE :  # 右眼，右眉
                                    x1 = np.int(landmark.x * w2)
                                    y1 = np.int(landmark.y * h2)
                                    right_eyes.append((x1, y1))
                                    eyes.append((x1, y1))
                            rotated_img, eye_center, angle=align_face(frame,right_iri,left_iri)
                            # cv2.imshow('eye_area',rotated_img)
                            rect = cv2.minAreaRect(np.array(eyes))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                            box = np.int0(box)
                            
                            print('*****',box)
                          
                            cv2.drawContours(frame, [box], 0, (255, 0, 0), 1)#画出眼睛区域
                            
                            # save_path=os.path.join('/data/cifs/f/Dataset/eyes_state/denger_lizhuo/1/','_'+img_name)
                            # cv2.imwrite(os.path.join('/data/cifs/f/Dataset/eyes_state/denger_lizhuo/1/','_'+image_name), img)
                            
                            
                            eye_area=imagecrop(rotated_img, box,image_name,angle,eye_center)#切出眼睛区域
                            cls_result=class_result(args,eye_area,image_name)
                            left_point_x,left_point_y,right_point_x,right_point_y=eye_state([box])
                            # cv2.rectangle(frame, (10,20), (50,60), (0,255,0), 4)   
                            cv2.putText(frame, cls_result, (left_point_x,left_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                            videowriter.write(frame)
                            # cv2.imwrite('output/result/'+image_name,frame)
                            '''
                            print(left_iri)
                            left_eye_states=eye_state(left_iri)
                            right_eye_states=eye_state(right_iri)
                            if left_eye_states=='close' and right_eye_states=='close':
                                save_img(img,left_eye_states,'/data/cifs/f/Dataset/eyes_state/denger_lizhuo/mediapipe_state/',image_name)
                            elif left_eye_states=='open' and right_eye_states=='open':
                                save_img(img,left_eye_states,'/data/cifs/f/Dataset/eyes_state/denger_lizhuo/mediapipe_state/',image_name)
                            # cv2.imshow('MediaPipe Face Mesh', img)
                            '''
                            if cv2.waitKey(5) & 0xFF == 27:
                                break
        else:
                print("所有帧都已经保存完成")
                break
    cap.release()            

if __name__ == "__main__":
    main()

