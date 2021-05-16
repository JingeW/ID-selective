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
# import cv2
import ssl

# turn certification check off
ssl._create_default_https_context = ssl._create_unverified_context


# Extractor for features
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for layer_name, module in self.submodule._modules.items():
            print(layer_name)
            if "classifier" in layer_name:
                x = x.view(x.size(0), -1)
            x = module(x)
            if self.extracted_layers is None or layer_name in self.extracted_layers and 'classifier' not in layer_name:
                outputs[layer_name] = x
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
    # print(net)
    # therd_size = 256
    extract_list = target_list
    transform = transforms.ToTensor()
    myextractor = FeatureExtractor(net, extract_list)

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
            features = v[0]  # squeeze the batch dim
            print(features.shape)
            iter_range = features.shape[0]
            print('iter_range: ', iter_range)
            for i in range(iter_range):
                # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
                if 'fc' in k:
                    continue

                feature = features.data.cpu().numpy()
                feature = feature[i, :, :]
                # feature_img = np.asarray(feature * 255, dtype=np.uint8)

                dst_path = os.path.join(dst, img_name, k)

                make_dirs(dst_path)
                # feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
                # if feature_img.shape[0] < therd_size:
                #     tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                #     tmp_img = feature_img.copy()
                #     tmp_img = cv2.resize(tmp_img, (therd_size,therd_size), interpolation =  cv2.INTER_NEAREST)
                #     cv2.imwrite(tmp_file, tmp_img)

                # dst_file = os.path.join(dst_path, str(i).zfill(3) + '.png')
                dst_matrix = os.path.join(dst_path, str(i).zfill(3) + '.csv')
                # cv2.imwrite(dst_file, feature_img)
                np.savetxt(dst_matrix, feature, delimiter=',')


if __name__ == '__main__':
    root = '/home/lab321/Jinge/ID_neuron_selectivity/CelebA_face_new'
    dest = '/home/lab321/Jinge/ID_neuron_selectivity/VGG16_ImgNet_feat_mat_full'
    path_list = [root + '/' + folder for folder in os.listdir(root)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = models.alexnet(pretrained=True, progress=True).features.to(device)
    model = models.vgg16(pretrained=True, progress=True).features.to(device)
    # print(model)
    target_list = None
    for path in path_list:
        ID = path.split('/')[-1]
        sub_dest = dest + '/' + ID
        get_feature(path, sub_dest, model, target_list)
