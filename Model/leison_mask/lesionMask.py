import numpy as np
import os
# import pickle


def randDrop_mask(layer, dim, drop_percent):
    length = dim * dim
    drops = int(length * drop_percent)
    # print(drops)
    mask = np.ones(length, dtype=int)
    mask[:drops] = 0
    np.random.shuffle(mask)
    mask = mask.reshape((dim, -1))
    print(mask.shape)
    return mask


convLayer_squence = [
    'Conv1_1', 'Conv1_2',
    'Conv2_1', 'Conv2_2',
    'Conv3_1', 'Conv3_2', 'Conv3_3',
    'Conv4_1', 'Conv4_2', 'Conv4_3',
    'Conv5_1', 'Conv5_2', 'Conv5_3',
]

layer_dim = {
    'Conv1_1': 224, 'Conv1_2': 224, 'Pool1': 112,
    'Conv2_1': 112, 'Conv2_2': 112, 'Pool2': 56,
    'Conv3_1': 56, 'Conv3_2': 56, 'Conv3_3': 56, 'Pool3': 28,
    'Conv4_1': 28, 'Conv4_2': 28, 'Conv4_3': 28, 'Pool4': 14,
    'Conv5_1': 14, 'Conv5_2': 14, 'Conv5_3': 14, 'Pool5': 7,
    'FC6': 4096, 'FC7': 4096, 'FC8': 2622,
}

drop_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for drop in drop_percent:
    mask_list = []
    for layer in convLayer_squence:
        dim = layer_dim[layer]
        mask = randDrop_mask(layer, dim, drop)
        mask_list.append(mask)

    save_dir = '/home/sdb1/Jinge/ID_selective/Model/leison_mask/masks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/randDrop_' + str(drop) + '.npy', 'wb') as f:
        np.save(f, mask_list)
