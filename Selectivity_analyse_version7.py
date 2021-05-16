'''Make random celebA 500 img full feat map set
'''
import os
import pickle
import numpy as np


def makeFullMatrix(layer, root, pkl_dir, sample_num):
    """ Method to group all feature maps from the same layer
        Parameters:
            layer: str, layer name
            root: str, path of the extraced feature
            pkl_dir: str, path to store the full matrix
            sample_num: int, number of sample per class
        Return:
            full_matrix: a matrix contain all the feature maps in the same player
                         each row is the flatten vector of one feature map(channel*w*h)
                         each col is the neuron response to the img (sample_num*class_num)
    """
    ID_list = sorted([root + '/' + f for f in os.listdir(root)])
    full_matrix = np.empty([0], dtype='float32')
    for ID in ID_list:
        img_list = sorted([ID + '/' + f for f in os.listdir(ID)])
        for img in img_list:
            # print('Now processing...', img)
            feat_list = sorted([img + '/' + layer + '/' + f for f in os.listdir(img + '/' + layer)])
            merge_feat = np.empty([0], dtype='float32')
            for feat in feat_list:
                f = open(feat, 'rb')
                matrix = pickle.load(f)
                f.close
                matrix = matrix.flatten()
                # print(matrix.dtype)
                merge_feat = np.concatenate((merge_feat, matrix))
                # print(matrix.dtype)
            full_matrix = np.concatenate((full_matrix, merge_feat))
    full_matrix = full_matrix.reshape((sample_num, -1))
    # print(full_matrix.shape)
    with open(pkl_dir + '/' + layer + '_fullMatrix.pkl', 'wb') as f:
        pickle.dump(full_matrix, f, protocol=4)


root = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_CelebA_original_featureMaps_randDrop_0.3_conv5_2'
pkl_dir = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_CelebA_original_randDrop_0.3_conv5_2'
if not os.path.exists(pkl_dir):
    os.makedirs(pkl_dir)
# layer_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_Vggface_random_cartoon_feat_mat_full_pkl/AbrahamLincoln/AbrahamLincoln0016'), reverse=True)
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
print('layer_list:', layer_list)
sample_num = 50 * 10
# sample_num = 2 * 35


for layer in layer_list:
    print('****************************')
    print('Building feature matrix of layer', layer + ':')
    makeFullMatrix(layer,  root,  pkl_dir,  sample_num)
