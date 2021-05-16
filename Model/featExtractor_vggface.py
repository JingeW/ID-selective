import os
import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
# from VGG16_vggface import VGG_16
# from VGG16_vggface_randDrop import VGG_16_perturb
from VGG16_vggface_randDrop_singleLayer import VGG_16_perturb_singleLayer
import pickle


def get_picture(pic_name, transform):
    print(pic_name)
    img = skimage.io.imread(pic_name)

    # img = skimage.io.imread(pic_name)[::-1, :]  # upside down

    # img = np.expand_dims(img, axis=2)  # 1 chanel to 3 chanels
    # img = np.concatenate((img, img, img), axis=-1)

    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def get_feature(root_dir, dst, model):
    net = model
    net.eval()
    transform = transforms.ToTensor()
    layer_name = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
        'FC6', 'FC7', 'FC8'
    ]

    pic_dir = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for pic in pic_dir:
        img = get_picture(pic, transform)
        img = img.unsqueeze(0)
        img = img.to(device)

        filename = os.path.split(pic)[1]
        img_name = os.path.splitext(filename)[0]
        print('----------', img_name, '----------')
        feat_list = net(img)
        for ind, feat in enumerate(feat_list):
            layer = layer_name[ind]
            print(layer)
            dst_path = os.path.join(dst, img_name, layer)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            if 'FC' in layer:
                features = feat
                feature = features.data.cpu().numpy()
                print(feature.shape)
                dst_matrix = os.path.join(dst_path, '0.pkl')
                with open(dst_matrix, 'wb') as f:
                    pickle.dump(feature, f, protocol=4)
            else:
                features = feat[0]
                print(features.shape)
                iter_range = features.shape[0]
                print('iter_range:', iter_range)
                for i in range(iter_range):
                    feature = features.data.cpu().numpy()
                    # print(feature.dtype)
                    feature = feature[i, :, :]
                    dst_matrix = os.path.join(dst_path, str(i).zfill(3) + '.pkl')
                    with open(dst_matrix, 'wb') as f:
                        pickle.dump(feature, f, protocol=4)


if __name__ == '__main__':
    root = '/home/sdb1/Jinge/ID_selective/Data/CelebA_original'
    dest = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_' + root.split('/')[-1] + '_featureMaps_randDrop_0.3_conv5_3'
    if not os.path.exists(dest):
        os.makedirs(dest)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_list = [root + '/' + folder for folder in os.listdir(root)]

    # model = VGG_16().to(device)
    # model = VGG_16_pertur.to(device)
    model = VGG_16_perturb_singleLayer().to(device)
    model.load_weights()
    # model.load_weights_layer_shuffle()
    # model.load_weights_kernel_shuffle()

    for path in path_list:
        ID = path.split('/')[-1]
        sub_dest = dest + '/' + ID
        get_feature(path, sub_dest, model)
