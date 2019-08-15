from yolo3.model import YoloBody, TinyYoloBody, model_saver, channel_move_forward, yolo_head
from yolo3.utils import letterbox_image
import os
import numpy as np
import torch
import colorsys
import time
import cv2
from PIL import Image, ImageFont, ImageDraw

class Yolo():
    def __init__(self):
        self.model_path = 'logs/veh_sign/Yolo_20180823-1623_11.100000381469727.pth' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/Sign_Veh_Navinfo_classes.txt'
        self.score = 0.1
        self.iou = 0.45
        self.model_image_size = (416, 416)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.device = torch.device("cuda")
        self.generate()
        self.detect_tools = DetectTools(self.anchors, self.class_names, self.model_image_size, self.score)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.pth'), 'Keras model or weights must be a .pkl file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        self.yolo_model = TinyYoloBody(num_anchors // 3, num_classes).to(self.device) \
            if is_tiny_version else YoloBody(num_anchors // 3, num_classes).to(self.device)
        model_saver(self.yolo_model, self.model_path, save=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        start = time.clock()
        '''
        image is  [1, 416, 416, 3], numpy.array
        output is [1, (anchors/3)*(classes+5), 416, 416]
        '''
        image_data = np.asarray(boxed_image)
        image_data = image_data[np.newaxis,:]
        inputs = channel_move_forward(image_data)
        print('time used',time.clock() - start)
        inputs = torch.Tensor(inputs).float().cuda()
        out_puts = self.yolo_model.forward(inputs)

        result = self.detect_tools.yolo_eval(out_puts, np.array([image.size[0], image.size[1]]))

        print('Found {} boxes for {}'.format(len(result), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 300
        thickness = 2
        for i, c in list(enumerate(result)):
            predicted_class = self.class_names[c[0]]
            box = c[1][1:]
            score = c[1][0]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c[0]])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c[0]])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image

class DetectTools():
    def __init__(self, anchors, class_names, model_image_size, score):
        self.anchors = anchors
        self.class_names = class_names
        self.model_image_size = model_image_size
        self.score = score

    def yolo_eval(self, out_puts, image_size):
        num_layers = len(out_puts)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
        Result = []
        for i in range(num_layers):
            box_xy, box_wh, box_confidence, box_class_probs = yolo_head(out_puts[i], self.anchors[anchor_mask[i]], len(self.class_names), self.model_image_size, 1,)
            box = self.yolo_boxes(box_xy[0], box_wh[0], np.array([self.model_image_size[1], self.model_image_size[0]]), image_size)
            box_score = box_confidence[0] * box_class_probs[0]
            # for x in range(shape[0]):
            #     for y in range(shape[2]):
            #         for z in range(shape[3]):
            #             BOXES.append(list(box[x,:,y,z]))
            #             SCORES.append(list(box_score[x,:,y,z]))
            box_re = np.reshape(box,[3, 4, -1])
            score_re = np.reshape(box_score, [3, len(self.class_names), -1])
            mask = score_re >= self.score
            mask_id = np.where(mask == True)
            score_result = score_re[mask_id]
            box_result = box_re[mask_id[0],:, mask_id[2]]
            for i in range(len(score_result)):
                Result.append([mask_id[1][i], score_result[i], box_result[i][0],box_result[i][1],box_result[i][2],box_result[i][3]])
        if len(Result) >= 1:
            result_to_show = self.non_max_suppression(np.array(Result))
        else:
            result_to_show = []
        return result_to_show

    def yolo_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''
        :param input_shape: must be numpy.array[col_num, row_num]
        :param image_shape: must be numpy.array[col_num, row_num]
        :return: box shape [3, 4, w, h] , is x_min, y_min, x_max, y_max
        '''
        new_shape = image_shape * np.min(input_shape / image_shape)
        scale = input_shape / new_shape
        offset = (input_shape - new_shape) / 2. /input_shape
        box_xy[:, 0, :, :] -= offset[0]
        box_xy[:, 1, :, :] -= offset[1]
        box_xy[:, 0, :, :] *= scale[0]
        box_xy[:, 1, :, :] *= scale[1]
        box_wh[:, 0, :, :] *= scale[0]
        box_wh[:, 1, :, :] *= scale[1]

        box_min = box_xy - (box_wh / 2)
        box_max = box_xy + (box_wh / 2)

        box = np.concatenate((box_min, box_max), 1)
        box_shape = np.shape(box)
        matrix = np.ones([3, 1, box_shape[2], box_shape[3]])
        image_shape_matrix = np.concatenate((matrix * image_shape[0], matrix * image_shape[1]),1)
        image_shape_matrix = np.concatenate((image_shape_matrix, image_shape_matrix),1)
        box *= image_shape_matrix
        return box

    def non_max_suppression(self,data, iou_thres = 0.45):
        result = []
        for i in range(len(self.class_names)):
            class_box = np.array(data[np.where(data[:,0] == i),1:][0])
            class_box_sort = class_box[np.argsort(-class_box[...,0])]
            for j in range(len(class_box_sort)):
                if len(class_box_sort) >= 2:
                    iou = self.iou(class_box_sort[1:, 1:], class_box_sort[0, 1:])
                    mask_iou = iou <= iou_thres
                    mask_id = np.insert(np.where(mask_iou == False)[0] + 1, 0,values=0)
                    result.append([i,class_box_sort[0]])
                    class_box_sort = np.delete(class_box_sort, mask_id,0)
                elif len(class_box_sort) == 0:
                    break
                elif len(class_box_sort) == 1:
                    result.append([i,class_box_sort.squeeze()])
                    break
        return result

    def iou(self,data,data_0):
        ixmin = np.maximum(data[:, 0], data_0[0])
        iymin = np.maximum(data[:, 1], data_0[1])
        ixmax = np.minimum(data[:, 2], data_0[2])
        iymax = np.minimum(data[:, 3], data_0[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        uni = ((data_0[2] - data_0[0] + 1.) * (data_0[3] - data_0[1] + 1.) +
               (data[:, 2] - data[:, 0] + 1.) *
               (data[:, 3] - data[:, 1] + 1.) - inters)
        iou = inters / uni
        return iou

def detect_image(yolo):
    path = 'F:/Navinfo/Object_detection/Traffic_Light_Detection/annotation/sign_vehicle_training_data/'
    i = 0
    while True:
        img = 'fc2_save_2018-07-12-173534-%s.png'%str(i).zfill(4)
        # img = input('Input image filename:')
        try:
            image = Image.open(path + img)
        except:
            print('Open Error! Try again!')
            i = int(i)
            i += 1
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            i = int(i)
            i += 1

if __name__ == '__main__':
    detect_image(Yolo())