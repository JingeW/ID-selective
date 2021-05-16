import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
# import matplotlib.pyplot as plt
import torchvision.models as models
# from PIL import Image


# Extractor for features
class FeatureExtractor_full(nn.Module):
    def __init__(self, model, extracted_layers):
        super(FeatureExtractor_full, self).__init__()
        self.model = model
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for layer_name, module in self.model._modules.items():
            if layer_name == 'classifier':
                # print('before:',x.shape)
                x = x.view(x.size(0), -1)
                # print('after:',x.shape)
            for sub_layer, submodule in module._modules.items():
                x = submodule(x)
                # print("classifier:", x.shape)
                if self.extracted_layers is None or sub_layer in self.extracted_layers:
                    outputs[layer_name + '_' + sub_layer.zfill(2)] = x
        return outputs


def get_picture(pic_name, transform):
    print(pic_name)
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(root_dir, dst, model, target_list):
    net = model
    net.eval()
    extract_list = target_list
    transform = transforms.ToTensor()
    myextractor = FeatureExtractor_full(net, extract_list)

    pic_dir = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for pic in pic_dir:
        img = get_picture(pic, transform)
        img = img.unsqueeze(0)  # insert the dim of batch = 1, since the input of cnn is a single img
        img = img.to(device)

        (filepath, filename) = os.path.split(pic)
        (img_name, extension) = os.path.splitext(filename)
        print('----------', img_name, '----------')
        outs = myextractor(img)

        for k, v in outs.items():  # k is layername, v is feature value with dim [batch, channel, w, h]
            print(k)
            dst_path = os.path.join(dst, img_name, k)
            make_dirs(dst_path)
            if k.split('_')[0] == 'classifier':
                features = v
                feature = features.data.cpu().numpy()
                print(features.shape)
                dst_matrix = os.path.join(dst_path, '0.csv')
                np.savetxt(dst_matrix, feature, delimiter=',')
            else:
                features = v[0]  # squeeze the batch dim
                print(features.shape)
                iter_range = features.shape[0]
                print('iter_range: ',  iter_range)
                for i in range(iter_range):
                    feature = features.data.cpu().numpy()
                    feature = feature[i, :, :]
                    dst_matrix = os.path.join(dst_path, str(i).zfill(3) + '.csv')
                    np.savetxt(dst_matrix, feature, delimiter=',')


if __name__ == '__main__':
    root = '/home/sdb1/Jinge/ID_selective/CelebA_face_new'
    dest = '/home/sdb1/Jinge/ID_selective/VGG16_ImgNet_feat_mat_full'
    path_list = [root + '/' + folder for folder in os.listdir(root)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True, progress=True).to(device)
    # print(model)
    target_list = None
    for path in path_list:
        ID = path.split('/')[-1]
        sub_dest = dest + '/' + ID
        get_feature(path, sub_dest, model, target_list)
