'''Comparison on Masks
'''
import os
import pickle
import numpy as np
from matplotlib_venn import venn2
from matplotlib import pyplot as plt


# def mask_comp(path1, path2, layer):
#     # load SNI
#     SNI_path1 = os.path.join(path1, layer + '_sig_neuron_ind.pkl')
#     with open(SNI_path1, 'rb') as f:
#         list1 = pickle.load(f)
#     SNI_path2 = os.path.join(path2, layer + '_sig_neuron_ind.pkl')
#     with open(SNI_path2, 'rb') as f:
#         list2 = pickle.load(f)
#     list_overlap = [value for value in list1 if value in list2]
#     return list_overlap


root = '/home/sdb1/Jinge/ID_selective/Result'
dest = '/home/sdb1/Jinge/ID_selective/Comparison'
save_path = '/home/sdb1/Jinge/ID_selective/Comparison/Comp_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
# layer_list = [
#     'Conv3_1', 'Conv3_2', 'Conv3_3',
#     'Conv5_1', 'Conv5_2', 'Conv5_3',
#     'FC6', 'FC7', 'FC8'
# ]

comparisons = [
    ('CelebA_original', 'CelebA_random'),
    # ('CelebA_original', 'CelebA_inverted'),
    # ('CelebA_original', 'CelebA_cartoon'), ('CelebA_original', 'cartoonFace_random'),
    # ('CelebA_random', 'CelebA_inverted'), ('CelebA_random', 'CelebA_cartoon'),
    # ('CelebA_random', 'cartoonFace_random'), ('CelebA_inverted', 'CelebA_cartoon'),
    # ('CelebA_inverted', 'cartoonFace_random'), ('CelebA_cartoon', 'cartoonFace_random')
]

for pair in comparisons:
    print(pair[0] + ' VS ' + pair[1])
    overlapPercent_dict = {}
    for layer in layer_list:
        print('------' + layer + '------')
        path1 = os.path.join(root, 'VGG16_Vggface_' + pair[0])
        path2 = os.path.join(root, 'VGG16_Vggface_' + pair[1])
        # list_overlap = mask_comp(path1, path2, layer)

        # print(pair[0], '_VS_', pair[1] + ':', str(len(list_overlap)) + 'overlaped')
        # save_dir = '/home/sdb1/Jinge/ID_selective/Comparison' + '/' + layer
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # with open(save_dir + '/' + pair[0] + 'vs' + pair[1] + '.pkl', 'wb') as f:
        #     pickle.dump(list_overlap, f)

        SNI_path1 = os.path.join(path1, layer + '_sig_neuron_ind.pkl')
        with open(SNI_path1, 'rb') as f:
            list1 = pickle.load(f)
        SNI_path2 = os.path.join(path2, layer + '_sig_neuron_ind.pkl')
        with open(SNI_path2, 'rb') as f:
            list2 = pickle.load(f)
        set_a = set(list1)
        set_b = set(list2)
        total = len(set_a.union(set_b))
        overlap = len(set_a.intersection(set_b))
        overlapPercent = np.divide(overlap, total) * 100
        overlapPercent_dict.update({layer: overlapPercent})

        plt.figure()
        v2 = venn2([set_a, set_b], set_labels=('', ''))
        v2.get_label_by_id('10').set_text('%s\n%d\n(%.0f%%)' % (pair[0],
                                                                len(set_a) - overlap,
                                                                np.divide(len(set_a) - overlap,
                                                                total) * 100))

        v2.get_label_by_id('01').set_text('%s\n%d\n(%.0f%%)' % (pair[1],
                                                                len(set_b) - overlap,
                                                                np.divide(len(set_b) - overlap,
                                                                total) * 100))

        v2.get_label_by_id('11').set_text('%s\n%d\n(%.0f%%)' % ('Overlap',
                                                                overlap,
                                                                overlapPercent))
        plt.title('Comparison of different masks of ' + layer)
        # save_dir = dest + '/' + layer
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        plt.savefig(save_path + '/' + pair[0] + 'vs' + pair[1] + '_' + layer + '.png', bbox_inches='tight', dpi=100)
        plt.savefig(save_path + '/' + pair[0] + 'vs' + pair[1] + '_' + layer + '.eps', format='eps', bbox_inches='tight', dpi=100)
        plt.savefig(save_path + '/' + pair[0] + 'vs' + pair[1] + '_' + layer + '.svg', format='svg', bbox_inches='tight', dpi=100)

    plt.figure()
    x = layer_list
    y = [overlapPercent_dict[k] for k in layer_list]
    a = plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.title('Overlap percentage between ' + pair[0] + ' and ' + pair[1])
    plt.savefig(save_path + '/' + pair[0] + '_vs_' + pair[1] + '.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + pair[0] + '_vs_' + pair[1] + '.eps', format='eps', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + pair[0] + '_vs_' + pair[1] + '.svg', format='svg', bbox_inches='tight', dpi=100)
