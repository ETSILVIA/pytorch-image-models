# import os
# import sys
# import torch

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))
# #from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
# from tinynn.converter import TFLiteConverter

# def main_worker():
#     model = torch.load('/workspace/pytorch-image-models/output/train/20230301-093157-mobilenetv3_large_100-128/checkpoint-62.pth')
#     print(model)
#     #model = model['model']
#     dummy_input = torch.randn(1, 3, 128, 128)
#     output_path = os.path.join(CURRENT_PATH,  'EYENET_model_epoch62_20230301.tflite')
#     # When converting quantized models, please ensure the quantization backend is set.
#     torch.backends.quantized.engine = 'qnnpack'
#     # The code section below is used to convert the model to the TFLite format
#     # If you want perform dynamic quantization on the float models,
#     # you may pass the following arguments.
#     #   `quantize_target_type='int8', hybrid_quantization_from_float=True, hybrid_per_channel=False`
#     converter = TFLiteConverter(model, dummy_input, output_path)
#     converter.convert()

# if __name__ == '__main__':
#     main_worker()

'''测试tiflite推理'''
# -*- coding:utf-8 -*-
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import time

import tensorflow as tf

test_image_dir = '/test_img/'
model_path = "out/EYENET_model_epoch62_20230301.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

# with tf.Session( ) as sess:
if 1:
    file_list = os.listdir(test_image_dir)

    model_interpreter_time = 0
    start_time = time.time()
    # 遍历文件
    for file in file_list:
        print('=========================')
        full_path = os.path.join(test_image_dir, file)
        print('full_path:{}'.format(full_path))

        img = cv2.imread(full_path)
        # 增加一个维度，变为 [1, 160, 160, 3]
        image_np_expanded = np.expand_dims(img, axis=0)
        image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求

        # 填装数据
        model_interpreter_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

        # 注意注意，我要调用模型了
        interpreter.invoke()
        output_data_age = interpreter.get_tensor(output_details[0]['index'])
        output_data_gender = interpreter.get_tensor(output_details[1]['index'])
        model_interpreter_time += time.time() - model_interpreter_start_time

        # 出来的结果去掉没用的维度
        Pre_age = 0
        Pre_gender = 0
        result_age = np.squeeze(output_data_age)
        print('Pre_age:{}'.format(result_age))
        # print('result:{}'.format(sess.run(output, feed_dict={newInput_X: image_np_expanded})))
        result_gender = np.squeeze(output_data_gender)
        print('Pre_gender:{}'.format(result_gender))

        # print('result:{}'.format((np.where(result == np.max(result)))[0][0]))
    used_time = time.time() - start_time
    print('used_time:{}'.format(used_time))
    print('model_interpreter_time:{}'.format(model_interpreter_time))