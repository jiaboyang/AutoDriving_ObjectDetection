
import numpy as np
from train import data_generator_wrapper
import torch
#
# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)
#
# with open('F:/Navinfo/Object_detection/Traffic_Light_Detection/annotation/train_Sign_Veh_Navinfo_crop3.txt') as f:
#     lines = f.readlines()
# num_train = len(lines)
#
# batch_size = 10
# input_shape = (416, 416)
# anchors = get_anchors('model_data/yolo_anchors.txt')
# num_classes = 6
# b = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
# # print(next(b))
# for i in b:
#     c,d = i[0],i[1:]
#     print(np.shape(i[0]))
#     print(np.shape(i[1]))
#     print(np.shape(i[2]))
#     print(np.shape(i[3]))
#     print(len(i))
#     print('#'*100)
#     # print(c)
#     # print(np.shape(np.array(c)),np.shape(np.array(d[0])))
#     print(np.shape(c),np.shape(d[0]))
#
# true_boxes = np.zeros(shape=[1,1,5])
#
# # a = preprocess_true_boxes([100,200,1.1,2.2,0],[13,13],anchors, 6)
# # print(a)

# a = np.ones(shape=[2,3,4])
#
# shape = [3,2,13,13]
# image = np.ones(shape, dtype=float)
#
#
# image_shape = np.array([3000,2000])
# input_shape = np.array([416, 416])
# new_shape = image_shape * np.min(input_shape / image_shape)
# scale = input_shape / new_shape
# image[:, 1, ...] *= scale[1]
# print(image)
# print(new_shape)
# print(scale)
# offset = (input_shape - new_shape) / 2. / input_shape
# print(offset)

a = 0.2

if a <= 0.03:
    print('haha')
elif a <= 0.1:
    print('hehe')
else:
    print('heihei')