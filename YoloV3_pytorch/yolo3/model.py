import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from yolo3.utils import compose
import collections.abc
import time
from torch.autograd import Variable

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bn=True, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else lambda x: x
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, out_channels):
        nn.Module.__init__(self)

        self.layer1 = Conv2d(out_channels, int(out_channels / 2), 1, padding=0)
        self.layer2 = Conv2d(int(out_channels / 2), out_channels, 3)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y + x
        return y

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,block_num):
        nn.Module.__init__(self)

        self.layer1 = Conv2d(in_channels,out_channels,3, stride=2)
        layers = []
        for _ in range(block_num):
            layers.append(BasicBlock(out_channels))
        self.layer2 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class DarknetBody(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.layers0 = Conv2d(3,32,3) #416
        self.layers1 = ResBlock(32,64,1) #208
        self.layers2 = ResBlock(64,128,2) #104
        self.layers3 = ResBlock(128,256,8) #52
        self.layers4 = ResBlock(256,512,8) #26
        self.layers5 = ResBlock(512,1024,4) #13

    def forward(self, x):
        x = self.layers0(x)
        x = self.layers1(x)
        x = self.layers2(x)
        y1 = self.layers3(x)
        y2 = self.layers4(y1)
        y3 = self.layers5(y2)
        return y1, y2, y3

class LastLayers(nn.Module):
    def __init__(self, in_channels, out_channels_x, out_channels_y):
        nn.Module.__init__(self)
        layers = []
        layers.append(Conv2d(in_channels, out_channels_x, 1, padding=0))
        layers.append(Conv2d(out_channels_x, out_channels_x*2, 3))
        layers.append(Conv2d(out_channels_x*2, out_channels_x, 1, padding=0))
        layers.append(Conv2d(out_channels_x, out_channels_x*2, 3))
        layers.append(Conv2d(out_channels_x*2, out_channels_x, 1,padding=0))
        self.layers_x = nn.Sequential(*layers)

        layers = []
        layers.append(Conv2d(out_channels_x, out_channels_x*2, 3))
        layers.append(Conv2d(out_channels_x*2,out_channels_y,1,padding=0, act=False))
        self.layers_y = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers_x(x)
        y = self.layers_y(x)
        return x, y

class YoloBody(nn.Module):
    def __init__(self, num_anchers, num_classes):
        nn.Module.__init__(self)

        self.darknet_body = DarknetBody()# 13 * 13 * 1024
        self.head_1 = LastLayers(1024, 512, num_anchers*(num_classes+5))# 13 * 13 * 512, 13 * 13 * y

        layers = []
        layers.append(Conv2d(512, 256, 1,padding=0))# 13 * 13 * 256
        layers.append(nn.Upsample(26,mode='nearest'))# 26 * 26 * 256
        self.head_2_layers0 = nn.Sequential(*layers)
        self.head_2_layers1 = LastLayers(768,256,num_anchers*(num_classes+5))

        layers = []
        layers.append(Conv2d(256,128,1,padding=0))
        layers.append(nn.Upsample(52,mode='nearest'))
        self.head_3_layers0 = nn.Sequential(*layers)
        self.head_3_layers1 = LastLayers(384,128,num_anchers*(num_classes+5))

    def forward(self, x):
        y1, y2, y3 = self.darknet_body(x)
        x, z3 = self.head_1(y3)

        x = self.head_2_layers0(x)
        x = torch.cat((x, y2), 1)
        x, z2 = self.head_2_layers1(x)

        x = self.head_3_layers0(x)
        x = torch.cat((x, y1), 1)
        x, z1 = self.head_3_layers1(x)
        return [z3, z2, z1]

class TinyYoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        nn.Module.__init__(self)


def model_saver(model, model_path = './logs/Navinfo_tl/Yolo.pkl', save = True, loss = 0):
    # 保存和加载整个模型
    # torch.save(model, './logs/veh_sign/params.pth')
    # model = torch.load('model.pkl')

    # 仅保存和加载模型参数(推荐使用)
    localtime = time.localtime()
    if save:
        # torch.save(model.state_dict(), './logs/veh_sign/Yolo_{}{}{}-{}{}{}_{}.pkl'.format(localtime[0],localtime[1],localtime[2],localtime[3],localtime[4],localtime[5], loss))
        torch.save(model.state_dict(),
                   './logs/Navinfo_tl/Yolo_{}{}{}-{}{}_{}.pth'.format(
                       localtime[0], str(localtime[1]).zfill(2), str(localtime[2]).zfill(2),
                       str(localtime[3]).zfill(2), str(localtime[4]).zfill(2), np.round(loss,1)))
    else:
        model.load_state_dict(torch.load(model_path))

def channel_move_forward(data):
    a = np.shape(data)
    if len(a) == 4:
        '''
        data is [batch, w, h, channel], np.array
        return is [batch, channel, w, h], np.array
        '''
        result = np.zeros(shape=[a[0], a[3], a[1], a[2]], dtype=float)
        for i in range(a[0]):
            for j in range(a[3]):
                result[i, j, ...] = data[i, ..., j]
        return result
    if len(a) == 5:
        '''
        data is [batch, w, h, channel0, channel1], np.array
        return is [batch, channel0, channel1, w, h], np.array
        '''
        result = np.zeros(shape=[a[0], a[3], a[4], a[1], a[2]], dtype=float)
        for i in range(a[0]):
            for j in range(a[3]):
                for k in range(a[4]):
                    result[i, j, k, ...] = data[i, ..., j, k]
        return result

def sigmoid(x):
    ex = np.exp(x)
    probs = ex / (ex + 1)
    return probs

def yolo_head(predict, anchors, num_classes, input_shape, batch_size, calc_loss = False):
    '''predict is [batch, w, h, channels]'''
    num_anchors = len(anchors)
    grid_shape = list(predict.size())[2:] # height , width
    grid_x = [list(range(0,grid_shape[1]))]*grid_shape[0]
    grid_y = [[list(range(0,grid_shape[0]))[i]]*grid_shape[1] for i in range(grid_shape[0])]
    grid_xy = np.array([[[grid_x, grid_y]] * num_anchors] * batch_size)

    grid_ones = np.ones(shape=grid_shape,dtype=float)
    grid_wh = np.array([list([[grid_ones * anchors[i][j] for j in range(len(anchors[0]))] for i in range(num_anchors)])] * batch_size)
    # print(np.shape(grid_wh))
    # print(grid_wh)
    predict = torch.Tensor.reshape(predict, [batch_size, num_anchors, num_classes+5, grid_shape[0], grid_shape[1]])

    output = predict.cpu().data.numpy()
    box_xy = (sigmoid(output[:,:,:2,:,:]) + grid_xy) / grid_shape[0] #这样求出来的是xy 在（0，1）范围[1, 3, 2, w, h]
    box_wh = np.exp(output[:,:,2:4,:,:]) * grid_wh / input_shape[0] #这样求出的是wh的长度，在（0，1）范围[1, 3, 2, w, h]
    box_confidence = sigmoid(output[:,:,4:5,:,:])# [1, 3, 1, w, h]
    box_class_prob = sigmoid(output[:,:,5:,:,:])# [1, 3, classes, w, h], 此处的wh是13/26/52

    if calc_loss == True:
        return grid_xy, grid_wh, predict, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_prob

def calc_ignore_mask(pred_box, true_box, object_mask, ignore_thresh):
    '''
    :param pred_box: a numpy, with shape [b, 3, 4, w, h], 4 is x, y, w, h
    :param ture_box: same as above
    :param object_mask: a numpy, with shape[b, 3, 1, w, h]
    :return:a numpy, with shape [b, 3, 1, w, h]
    '''
    ignore_mask = np.ones(shape=np.shape(object_mask),dtype=float)
    mask_id = np.where(object_mask == 1)
    true_box = true_box[mask_id[0], mask_id[1], :, mask_id[3], mask_id[4]]
    if len(mask_id[0]) > 0:
        set_batch = set(mask_id[0])
        for i in set_batch:
            true_box_i = true_box[np.where(mask_id[0] == i)]
            iou = box_iou(pred_box[i], true_box_i)
            best_iou = np.amax(iou, axis=0)
            ignore_mask_i = np.asarray(best_iou < ignore_thresh).astype(np.int)
            ignore_mask[i] = ignore_mask_i
    return ignore_mask

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    b1: numpy, shape=(3,4,w,h), xywh
    b2: numpy, shape=(j, 4), xywh
    Returns
    iou: tensor, shape=(j,3,1,w,h)
    '''

    # Expand dim to apply broadcasting.
    b1 = np.expand_dims(b1, 0)
    b1_xy = b1[:, :, :2, :, :]
    b1_wh = b1[:, :, 2:4,:, :]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # Expand dim to apply broadcasting.
    b2 = np.expand_dims(np.expand_dims(np.expand_dims(b2, 1),-1),-1)
    b2_xy = b2[:, :, :2, :, :]
    b2_wh = b2[:, :, 2:4, :, :]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)#[j,3,2,w,h]
    intersect_area = intersect_wh[:, :, 0,:,:] * intersect_wh[:,:, 1,:,:]#[j,3,w,h]
    b1_area = b1_wh[:,:, 0,:,:] * b1_wh[:,:, 1,:,:]#[1,3,w,h]
    b2_area = b2_wh[:,:, 0,:,:] * b2_wh[:,:, 1,:,:]#[j,1,1,1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)#[j,3,w,h]
    iou = np.expand_dims(iou, 2)
    return iou

def yolo_loss(predict, label, anchors, num_classes, ignore_thresh = .5, print_loss = True):
    '''
    predict is [[batch, channels, w, h],[],[]]
    label is [[batch, num_layers, 5+classes, w, h],[],[]]
    '''
    num_layers = len(anchors) // 3
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers ==3 else [[3,4,5],[0,1,2]]
    input_shape = [list(predict[0].size())[i] * 32 for i in range(2,4)]
    grid_shape = [list(predict[i].size())[2:] for i in range(num_layers)]
    loss = 0
    batch_size = list(predict[0].size())[0]

    # print(input_shape)
    # print(grid_shape)
    mseloss = nn.MSELoss(reduce=False)
    bcelogitsloss = nn.BCEWithLogitsLoss(reduce=False)

    for i in range(num_layers):
        # print(label[i])
        object_mask = torch.Tensor(label[i][:,:,4:5,:,:]).cuda()
        true_class_probs = torch.Tensor(label[i][:,:,5:,:,:]).cuda()
        grid_xy, grid_wh, raw_predict, pred_xy, pred_wh = yolo_head(predict[i], anchors[anchor_mask[i]], num_classes, input_shape, batch_size, calc_loss=True)

        raw_true_xy = np.clip(label[i][:,:,:2,:,:] * grid_shape[i][0] - grid_xy, 0.,1.)
        raw_true_wh = input_shape[0] * label[i][:,:,2:4,:,:] / grid_wh
        raw_true_wh = np.select([raw_true_wh != 0], [np.log(raw_true_wh)])# avoid log(0)=-inf
        box_loss_scale = 2.

        pred_box = np.concatenate((pred_xy, pred_wh),2)
        true_box = np.concatenate((raw_true_xy, raw_true_wh),2)
        ignore_mask = calc_ignore_mask(pred_box, true_box, label[i][:,:,4:5,:,:], ignore_thresh)
        raw_true_xy = torch.Tensor(raw_true_xy).cuda()
        raw_true_wh = torch.Tensor(raw_true_wh).cuda()
        ignore_mask = torch.Tensor(ignore_mask).cuda()

        wh_loss = torch.sum(mseloss(raw_predict[:,:,2:4,:,:], raw_true_wh) * object_mask * box_loss_scale * 0.5)
        ''' 写成如下这样也不犯法
        # object_mask_complement = torch.Tensor(1 - label[i][:,:,4:5,:,:]).expand(1,batch_size, 3, 1, grid_shape[i][0], grid_shape[i][1])
        #object_mask_expand = object_mask.expand(1, batch_size, 3, 1, grid_shape[i][0], grid_shape[i][1])
        # confidence_loss = bceloss(raw_predict[:, :, 4:5, :, :], object_mask)
        # confidence_loss = torch.sum(torch.prod(torch.cat((confidence_loss.expand(1,batch_size, 3, 1, grid_shape[i][0], grid_shape[i][1]), object_mask_expand),0),0)) + torch.sum(torch.prod(torch.cat((confidence_loss.expand(1, batch_size, 3, 1, grid_shape[i][0], grid_shape[i][1]), object_mask_complement), 0),0))
        '''
        xy_loss = torch.sum(bcelogitsloss(raw_predict[:,:,:2,:,:], raw_true_xy) * object_mask * box_loss_scale)
        confidence_loss = torch.sum(bcelogitsloss(raw_predict[:,:,4:5,:,:], object_mask) * object_mask) + torch.sum(bcelogitsloss(raw_predict[:,:,4:5,:,:], object_mask) * (1 - object_mask) * ignore_mask)
        class_loss = torch.sum(bcelogitsloss(raw_predict[:,:,5:,:,:], true_class_probs) * object_mask)#需要说明的是shape不相等，但是维度相等就可以

        xy_loss = xy_loss / batch_size
        wh_loss = wh_loss / batch_size
        confidence_loss = confidence_loss / batch_size
        class_loss = class_loss / batch_size

        loss += xy_loss +wh_loss + confidence_loss +class_loss
    if print_loss:
        print(loss)
    return loss