import os
import pickle
# import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


def makeFolder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def makeLabels(sample_num, class_num):
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label


# *********Adjustable***********
dataSet_name = 'CelebA_original'
sample_num = 10
class_num = 50
# ******************************

FM_dir = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_' + dataSet_name
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
result_dir = '/home/sdb1/Jinge/ID_selective/Result/Tsne/' + dataSet_name
label = makeLabels(sample_num, class_num)
layer_list = [
    # 'Conv1_1', 'Conv1_2', 'Pool1',
    # 'Conv2_1', 'Conv2_2', 'Pool2',
    # 'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    # 'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    # 'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    # 'FC6', 'FC7', 'FC8'
    'Conv5_3', 'FC6'
]

for layer in layer_list:
    print('****************************')
    print('Now working on layer:', layer)

    print('Loading feature matrix...')
    with open(FM_dir + '/' + layer + '_fullMatrix.pkl', 'rb') as f:
        fullMatrix = pickle.load(f)

    print('Loading mask...')
    with open(mask_dir + '/' + layer + '_sig_neuron_ind.pkl', 'rb') as f:
        maskID = pickle.load(f)
    with open(mask_dir + '/' + layer + '_non_sig_neuron_ind.pkl', 'rb') as f:
        maskNonID = pickle.load(f)
    print('Length of ID/nonID mask:', len(maskID), len(maskNonID))

    perplexity_ID = min(math.sqrt(len(maskID)), 500)
    perplexity_nonID = min(math.sqrt(len(maskNonID)), 500)
    print('perplexity:', perplexity_ID, perplexity_nonID)

    valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
    markers = valid_markers + valid_markers[:class_num - len(valid_markers)]

    tsne_ID = TSNE(perplexity=perplexity_ID).fit_transform(fullMatrix[:, maskID])

    save_path = result_dir + '/' + layer
    makeFolder(save_path)
    plt.figure()
    for i in range(50):
        plt.scatter(tsne_ID[i * sample_num: i * sample_num + sample_num, 0],
                    tsne_ID[i * sample_num: i * sample_num + sample_num, 1],
                    label[i * sample_num: i * sample_num + sample_num], marker=markers[i])
    plt.title(dataSet_name + ' ' + layer + '_ID')
    plt.savefig(save_path + '/tsne_ID.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/tsne_ID.eps', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/tsne_ID.svg', bbox_inches='tight', dpi=100)

    tsne_nonID = TSNE(perplexity=perplexity_nonID).fit_transform(fullMatrix[:, maskNonID])
    plt.figure()
    for i in range(50):
        plt.scatter(tsne_nonID[i * sample_num: i * sample_num + sample_num, 0],
                    tsne_nonID[i * sample_num: i * sample_num + sample_num, 1],
                    label[i * sample_num: i * sample_num + sample_num], marker=markers[i])
    plt.title(dataSet_name + ' ' + layer + '_nonID')
    plt.savefig(save_path + '/tsne_nonID.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/tsne_nonID.eps', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/tsne_nonID.svg', bbox_inches='tight', dpi=100)
