''' Visualize the feature maps of different channel from each conv layer
    Apply mask on the feature maps
    Compare Full/ID/nonID using visual judgment
    *** randomly picked 10(5*2) images
'''

import os
import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
from VGG16_vggface import VGG_16
import pickle
import matplotlib
import random


def mask_registration(fm_size, channel_size, mask):
    IDmask_list = []
    nonIDmask_list = []
    mask_template = np.zeros(fm_size)
    for i in range(channel_size):
        imask = [item for item in mask if item > fm_size * i and item < fm_size * (i + 1)]
        imask = [item - fm_size * i for item in imask]
        IDmask = np.array([(i in imask) for i in range(len(mask_template))])
        nonIDmask = [not item for item in IDmask]
        IDmask = np.array(list(map(int, IDmask)))
        nonIDmask = np.array(list(map(int, nonIDmask)))
        IDmask = IDmask.reshape((int(np.sqrt(fm_size)), -1))
        nonIDmask = nonIDmask.reshape((int(np.sqrt(fm_size)), -1))
        IDmask_list.append(IDmask)
        nonIDmask_list.append(nonIDmask)
    return IDmask_list, nonIDmask_list


def get_picture(pic_name, transform):
    print(pic_name)
    img = skimage.io.imread(pic_name)
    print(img.size())

    # img = skimage.io.imread(pic_name)[::-1, :]  # upside down

    img = np.expand_dims(img, axis=2)  # 1 chanel to 3 chanels
    img = np.concatenate((img, img, img), axis=-1)
    print(img.size())

    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def get_feature(root_dir, dst, model):
    net = model
    net.eval()
    transform = transforms.ToTensor()
    sig_ind_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
    layer_name = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    ]

    pic_dir = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)])
    randPic_dir = random.sample(pic_dir, 2)  # Randomly pick 2 pic to form the pic_dir
    for pic in randPic_dir:
        img = get_picture(pic, transform)
        img = img.unsqueeze(0)
        img = img.to(device)

        filename = os.path.split(pic)[1]
        img_name = os.path.splitext(filename)[0]
        print('----------', img_name, '----------')
        feat_list = net(img)
        for ind, feat in enumerate(feat_list):
            layer = layer_name[ind]
            print(ind, 'Layer:', layer)
            dst_path = os.path.join(dst, img_name, layer)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            features = feat[0]
            print('Shape of feature map', features.shape)
            channel_size = features.shape[0]
            fm_size = features.shape[1] * features.shape[2]
            print('channel_size:', channel_size, 'fm_size:', fm_size)

            # load mask
            print('Loading mask...')
            SNI_path = os.path.join(sig_ind_dir, layer + '_sig_neuron_ind.pkl')
            f = open(SNI_path, 'rb')
            mask = pickle.load(f)
            f.close
            IDmask_list, nonIDmask_list = mask_registration(fm_size, channel_size, mask)
            del mask
            print('Mask loading complete, memory released!')

            # extract feature and save as image
            for i in range(channel_size):
                feature = features.data.cpu().numpy()
                feature = feature[i, :, :]
                IDmask = IDmask_list[i]
                nonIDmask = nonIDmask_list[i]
                # # Save feature map as matrix
                # dst_matrix = os.path.join(dst_path, str(i).zfill(3) + '.pkl')
                # with open(dst_matrix, 'wb') as f:
                #     pickle.dump(feature, f, protocol=4)

                # Save feature map as img
                dst_image = os.path.join(dst_path, str(i).zfill(3) + '_Full.png')
                matplotlib.image.imsave(dst_image, feature)

                dst_IDimage = os.path.join(dst_path, str(i).zfill(3) + '_ID.png')
                IDfeature = feature * IDmask
                matplotlib.image.imsave(dst_IDimage, IDfeature)

                dst_nonIDimage = os.path.join(dst_path, str(i).zfill(3) + '_nonID.png')
                nonIDfeature = feature * nonIDmask
                matplotlib.image.imsave(dst_nonIDimage, nonIDfeature)
            del IDmask_list, nonIDmask_list
            print('Feature map of layer:', layer, 'saved, memory released!')


if __name__ == '__main__':
    root = '/home/sdb1/Jinge/ID_selective/Data/CelebA_original'
    dest = '/home/sdb1/Jinge/ID_selective/Result/Visualize'
    if not os.path.exists(dest):
        os.makedirs(dest)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    path_list = sorted([root + '/' + folder for folder in os.listdir(root)])
    randPath_list = random.sample(path_list, 5)  # Randomly pick 5 IDs
    model = VGG_16().to(device)
    model.load_weights()

    for path in randPath_list:
        ID = path.split('/')[-1]
        sub_dest = dest + '/' + ID
        get_feature(path, sub_dest, model)

    result_list = sorted([os.path.join(dest, folder) for folder in os.listdir(dest)])
    for result in result_list:
        img_list = sorted([os.path.join(result, folder) for folder in os.listdir(result)])
        for img in img_list:
            layer_list = sorted([os.path.join(img, folder) for folder in os.listdir(img)])
            for layer in layer_list:
                item_list = sorted([os.path.join(layer, folder) for folder in os.listdir(layer)])
                for item in item_list:
                    featmap = skimage.io.imread(item)
                    if not featmap.shape[0] == 224:
                        featmap = skimage.transform.resize(featmap, (224, 224))
                        token = item.split('/')
                        token[-1] = token[-1].split('.')[0] + '_resized.png'
                        save_path = '/'.join(token)
                        matplotlib.image.imsave(save_path, featmap)
