''' Visualize the feature maps of different channel from each conv layer
    Apply mask on the feature maps
    Compare Full/ID/nonID using visual judgment
    *** Input is 1 selected image
        Mask registration updated
'''

import os
import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
from VGG16_vggface import VGG_16
# from VGG16_vggface_randDrop import VGG_16_perturb
import pickle
import matplotlib


def mask_registration(fm_size, channel_size, mask):
    FM_mask_list = []
    mask_template = np.zeros(fm_size)
    for i in range(channel_size):
        FM_mask = [item for item in mask if item >= fm_size * i and item < fm_size * (i + 1)]  # Pick slice
        FM_mask = [item - fm_size * i for item in FM_mask]  # Normalize the index
        FM_mask = np.array([(i in FM_mask) for i in range(len(mask_template))])  # Binary encoding
        FM_mask = np.array(list(map(int, FM_mask)))  # Binary to int
        FM_mask = FM_mask.reshape((int(np.sqrt(fm_size)), -1))  # Resize 2D
        FM_mask_list.append(FM_mask)
    return FM_mask_list


def get_picture(pic_name, transform):
    print(pic_name)
    img = skimage.io.imread(pic_name)

    # img = skimage.io.imread(pic_name)[::-1, :]  # upside down

    # img = np.expand_dims(img, axis=2)  # 1 chanel to 3 chanels
    # img = np.concatenate((img, img, img), axis=-1)

    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def get_feature(pic_path, dst, model):
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

    img = get_picture(pic_path, transform)
    img = img.unsqueeze(0)
    img = img.to(device)

    filename = os.path.split(pic_path)[1]
    img_name = os.path.splitext(filename)[0]
    print('----------', img_name, '----------')
    feat_list = net(img)
    for ind, feat in enumerate(feat_list):
        if ind < len(layer_name):
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
            IDmask_path = os.path.join(sig_ind_dir, layer + '_sig_neuron_ind.pkl')
            with open(IDmask_path, 'rb') as f:
                IDmask = pickle.load(f)
            IDmask_list = mask_registration(fm_size, channel_size, IDmask)

            nonIDmask_path = os.path.join(sig_ind_dir, layer + '_non_sig_neuron_ind.pkl')
            with open(nonIDmask_path, 'rb') as f:
                nonIDmask = pickle.load(f)
            nonIDmask_list = mask_registration(fm_size, channel_size, nonIDmask)
            print('Mask loading complete.')

            # extract feature and save as image
            for i in range(channel_size):
                feature = features.data.cpu().numpy()
                feature = feature[i, :, :]
                IDmask = IDmask_list[i]
                nonIDmask = nonIDmask_list[i]

                # Save feature map as img
                dst_image = os.path.join(dst_path, str(i).zfill(3) + '_Full.png')
                matplotlib.image.imsave(dst_image, feature)

                dst_IDimage = os.path.join(dst_path, str(i).zfill(3) + '_ID.png')
                IDfeature = feature * IDmask
                matplotlib.image.imsave(dst_IDimage, IDfeature)

                dst_nonIDimage = os.path.join(dst_path, str(i).zfill(3) + '_nonID.png')
                nonIDfeature = feature * nonIDmask
                matplotlib.image.imsave(dst_nonIDimage, nonIDfeature)
            print('Feature map of layer:', layer, 'saved!')


if __name__ == '__main__':
    pic_path = '/home/sdb1/Jinge/ID_selective/Data/CelebA_original/10/075150.jpg'
    # dest = '/home/sdb1/Jinge/ID_selective/Result/Visualize/' + pic_path.split('/')[-4]
    dest = '/home/sdb1/Jinge/ID_selective/Result/Visualize/CelebA_original_sample'
    if not os.path.exists(dest):
        os.makedirs(dest)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG_16().to(device)
    # model = VGG_16_perturb().to(device)
    model.load_weights()

    get_feature(pic_path, dest, model)

    img_list = sorted([os.path.join(dest, folder) for folder in os.listdir(dest)])
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
