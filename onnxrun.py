import onnxruntime
import numpy as np
import cv2
import math
import torch
from PIL import Image
from qua_mobilenetv3 import mobilenet_v3_large 
from torchvision.transforms import transforms
img_path = 'data/cifs/f/Dataset/eyes_state/003_new_data/train_2022-12-13/no_glass/open/0001_chouyan_nafang_022_4.jpg'
MODEL_PATH = 'mbnv3_large_20221230.onnx'
eye_states=['open','close','invisible']
glass_states=['no_glass','glass','sunglass']

def softmax(x):
    tmp=np.max(x)
    x-=tmp
    x=np.exp(x)
    tmp=np.sum(x)
    x/=tmp
    return x

def main():
    val_transform = transforms.Compose([transforms.Resize((128, 128)),

                            
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        
                                        ])
    session = onnxruntime.InferenceSession(MODEL_PATH)
    input_details = session.get_inputs()
    output_details = session.get_outputs()
    input_ = input_details[0]
    assert input_.shape == [1, 3, 128, 128]
    assert input_.name == 'input.1'

    # im=Image.open(img_path).convert('RGB')
    # im_rgb=transforms.Resize
    # print(im_rgb.size())
    # im_rgb= torch.unsqueeze(im_rgb, dim=0)
    im = cv2.imread(img_path)
    # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_rgb=cv2.resize(im,(128,128))

    
    im_rgb = im_rgb.astype('float32')
    im_rgb = im_rgb/255

    print(im_rgb.shape)
    im_rgb = im_rgb[np.newaxis, :].transpose([0, 3, 1, 2])
    
    print(im_rgb.shape)

    outputs = ["652","651"]
    #glass 548,eye 549
    result = session.run(outputs, {'input.1': im_rgb})
    print(type(result), len(result))
    # glass_result=softmax(result[1][0])
    # eye_result=softmax(result[0][0])
    glass_result=result[1][0]
    eye_result=result[0][0]
    print(eye_result)
    glass_value = np.argmax(glass_result) 
    glass_state = glass_states[glass_value]
    
    eye_value = np.argmax(eye_result) 
    eye_state = eye_states[eye_value]
    
    print(glass_state)
    print(eye_state)
    

if __name__ == '__main__':
    main()