from yolo3.model import YoloBody,TinyYoloBody, yolo_loss, model_saver, channel_move_forward
from yolo3.utils import get_random_data
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class TrainYolo3():
    def __init__(self):
        self.annotation_path = 'F:/Navinfo/Object_detection/Traffic_Light_Detection/annotation/train_TL_Navinfo_crop5.txt'  # label saved
        self.log_dir = 'logs/Navinfo_tl/'  # model saved
        classes_path = 'model_data/TL_Navinfo_classes.txt'  # class saved
        anchors_path = 'model_data/yolo_anchors.txt'  # anchors saved
        class_names = get_classes(classes_path)
        self.num_classes = len(class_names)
        self.anchors = get_anchors(anchors_path)  # 9*2

        self.input_shape = (416, 416)  # multiple of 32, hw
        device = torch.device("cuda")
        is_tiny_version = len(self.anchors) == 6  # default setting
        if is_tiny_version:
            self.model = create_tiny_model(self.anchors, self.num_classes, device,weights_path='model_data/yolo-tiny.h5')
        else:
            self.model = create_model(self.anchors, self.num_classes, device, load_pretrained = True, weights_path='./logs/Navinfo_tl/Yolo_20180817-1815_5.900000095367432.pth')
        self.train()

    def train(self):
        val_split = 0.1
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        # for p in self.model.darknet_body.parameters():
        #     p.requires_grad = False
        # print('Freeze the darknet layers')
        # batch_size = 10
        # for _ in range(2):
        #     self.train_process(data_generator_wrapper(lines[:num_train], batch_size, self.input_shape, self.anchors, self.num_classes),
        #               num_train, batch_size, learning_rate=1e-3)

        for p in self.model.darknet_body.parameters():
            p.requires_grad = True
        print('Unfreeze all the layers')
        batch_size = 10
        loss_last = 391.540
        learning_rate = 1e-6
        for _ in range(100):
            self.train_process(data_generator_wrapper(lines[:num_train], batch_size, self.input_shape, self.anchors, self.num_classes),
                          num_train, batch_size, learning_rate=learning_rate)
            valid_loss = self.validation_process(data_generator_wrapper(lines[num_train:], batch_size, self.input_shape, self.anchors, self.num_classes),
                          batch_size)
            if valid_loss >= loss_last:
                learning_rate *= 0.1
                print('Train next epoch on learning rate {}'.format(learning_rate))
            loss_last = valid_loss

    def train_process(self,train_data, num_train, batch_size, learning_rate):
        i = 0
        loss_list = []
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for batch_data in train_data:
            print('Train {} out of {}'.format(i, num_train))
            optimizer.zero_grad()
            inputs, label = batch_data[0], batch_data[1:]
            inputs = channel_move_forward(inputs)
            label = [channel_move_forward(label[i]) for i in range(3)]
            inputs = torch.Tensor(inputs).float().cuda()
            outputs = self.model(inputs)
            loss = yolo_loss(outputs, label, self.anchors, self.num_classes)
            loss_list.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            i += batch_size
            if i >= num_train:
                model_saver(self.model, loss=np.mean(loss_list))
                break

    def validation_process(self, train_data, batch_size):
        loss_val = []
        i = 0
        for batch_data in train_data:
            inputs, label = batch_data[0], batch_data[1:]
            inputs = channel_move_forward(inputs)
            label = [channel_move_forward(label[i]) for i in range(3)]
            inputs = torch.Tensor(inputs).float().cuda()
            outputs = self.model(inputs)
            loss = yolo_loss(outputs, label, self.anchors, self.num_classes, print_loss=False)
            loss_val.append(loss.cpu().data.numpy())
            i += batch_size
            if i >= 1000:
                lossmean = np.mean(loss_val)
                print('validation loss:', lossmean)
                break
        return lossmean

'''
The followings are the helper functions, which do not containing adjustable parameters.
'''

def create_tiny_model(anchors, num_classes, device, load_pretrained=True,weights_path='model_data/yolo_weights.h5'):
    model = 0
    return model

def create_model(anchors, num_classes, device, load_pretrained=True,weights_path='./logs/veh_sign/Yolo.pkl'):
    '''create the training model'''
    num_anchors = len(anchors)

    model_body = YoloBody(num_anchors // 3, num_classes)#.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model_body = nn.DataParallel(model_body)
    if torch.cuda.is_available():
        model_body.to(device)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_saver(model_body, weights_path, save=False)
        print('Load weights')
    return model_body

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true]

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    '''
    return shape:
    for i in result:
        i is a list
        i[0] is image with  [batch_size, 416, 416, 3], numpy.array
        i[1] is label with  [batch_size, 13, 13, 3 (3 anchors in each head), 5+classes], numpy.array
        i[2] is label with  [batch_size, 26, 26, 3 (3 anchors in each head), 5+classes], numpy.array
        i[3] is label with  [batch_size, 52, 52, 3 (3 anchors in each head), 5+classes], numpy.array (no if tiny)
    '''
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
    return y_true

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    TrainYolo3()