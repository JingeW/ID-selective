# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchfile
import numpy as np
import skimage.data
import skimage.io
import skimage.transform


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        # self.fc8 = nn.Linear(4096, 2)

    def load_weights(self, path='/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    # print(self_layer)
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    # print(self_layer.weight.data.shape)
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    # print(self_layer.weight.data.shape)  # [out, in, h, w]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                # else:
                #     if not block == 8:
                #         self_layer = getattr(self, "fc%d" % (block))
                #         block += 1
                #         self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                #         self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def load_weights_layer_shuffle(self, path='/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, 'conv_%d_%d' % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1

                    # shuffle layer
                    weight_f = layer.weight.flatten()
                    np.random.shuffle(weight_f)
                    weight = weight_f.reshape(layer.weight.shape)

                    self_layer.weight.data[...] = torch.tensor(weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    for row in range(layer.weight.shape[0]):
                        np.random.shuffle(layer.weight[row])
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def load_weights_kernel_shuffle(self, path='/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, 'conv_%d_%d' % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1

                    # shuffle kernel
                    for i in range(layer.weight.shape[0]):
                        for j in range(0, layer.weight.shape[1], 9):
                            np.random.shuffle(layer.weight[i, :][j: j + 9])

                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    for row in range(layer.weight.shape[0]):
                        np.random.shuffle(layer.weight[row])
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x01 = F.relu(self.conv_1_1(x))
        # print('x01size:', x01.shape[2],x01.shape[3])
        x02 = F.relu(self.conv_1_2(x01))
        x03 = F.max_pool2d(x02, 2, 2)
        x04 = F.relu(self.conv_2_1(x03))
        x05 = F.relu(self.conv_2_2(x04))
        x06 = F.max_pool2d(x05, 2, 2)
        x07 = F.relu(self.conv_3_1(x06))
        x08 = F.relu(self.conv_3_2(x07))
        x09 = F.relu(self.conv_3_3(x08))
        x10 = F.max_pool2d(x09, 2, 2)
        x11 = F.relu(self.conv_4_1(x10))
        x12 = F.relu(self.conv_4_2(x11))
        x13 = F.relu(self.conv_4_3(x12))
        x14 = F.max_pool2d(x13, 2, 2)
        x15 = F.relu(self.conv_5_1(x14))
        x16 = F.relu(self.conv_5_2(x15))
        x17 = F.relu(self.conv_5_3(x16))
        x18 = F.max_pool2d(x17, 2, 2)
        x19 = x18.view(x18.size(0), -1)
        x20 = F.relu(self.fc6(x19))
        x21 = F.dropout(x20, 0.5, self.training)
        x22 = F.relu(self.fc7(x21))
        x23 = F.dropout(x22, 0.5, self.training)

        x24 = F.relu(self.fc8(x23))
        feat = [
            x01, x02, x03,
            x04, x05, x06,
            x07, x08, x09, x10,
            x11, x12, x13, x14,
            x15, x16, x17, x18,
            x20, x22, x24
        ]
        return feat

        # return self.fc8(x23)


def get_picture(pic_name, transform):
    print(pic_name)
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


if __name__ == "__main__":
    model = VGG_16()
    model.load_weights()
    transform = transforms.ToTensor()
    im = get_picture("/home/sdb1/Jinge/ID_selective/Data/CelebA_original/02/002582.jpg", transform)
    im = im.unsqueeze(0)

    model.eval()
    # feat_list = model(im)
    # print(len(feat_list))
    pred = F.softmax(model(im))
    values, indices = pred.max(-1)
