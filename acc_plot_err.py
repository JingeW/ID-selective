'''Plot acc with error shade
    11 inputs intotal:
    CelebA_original
    CelebA_random
    CelebA_inverted
    CelebA_MooneyFace
    CelebA_cartoon_Hayao
    CelebA_cartoon_Hosoda
    CelebA_cartoon_Paprika
    CelebA_cartoon_Shinkai
    cartoonFace_random
    ImageNet_random
    localizer_balanced
'''
import pickle
from matplotlib import pyplot as plt

name_list = [
    'CelebA_original',
    # 'CelebA_random',
    # 'CelebA_inverted',
    # 'CelebA_MooneyFace',
    # 'CelebA_cartoon_Hayao',
    # 'CelebA_cartoon_Hosoda',
    # 'CelebA_cartoon_Paprika',
    # 'CelebA_cartoon_Shinkai',
    # 'cartoonFace_random',
    # 'ImageNet_random',
    # 'localizer_balanced',
    # 'CelebA_original_layer_shuffle'
    # 'CelebA_original_kernel_shuffle'
    # 'CelebA_original_randDrop_0.3_conv1_1'
]
root = '/home/sdb1/Jinge/ID_selective/kFolder/'
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
for dataSet_name in name_list:
    # dataSet_name = 'cartoonFace_random'
    acc_dir = root + dataSet_name

    with open(acc_dir + '/full_acc_dict.pkl', 'rb') as f:
        full_acc_dict = pickle.load(f)
    with open(acc_dir + '/ID_acc_dict.pkl', 'rb') as f:
        ID_acc_dict = pickle.load(f)
    with open(acc_dir + '/nonID_acc_dict.pkl', 'rb') as f:
        nonID_acc_dict = pickle.load(f)
    # with open(acc_dir + '/SIMI_acc_dict.pkl', 'rb') as f:
    #     SIMI_acc_dict = pickle.load(f)
    # with open(acc_dir + '/overlap_acc_dict.pkl', 'rb') as f:
    #     overlap_acc_dict = pickle.load(f)

    x = layer_list
    y = [[full_acc_dict[k].mean(), full_acc_dict[k].std()] for k in layer_list]
    y1 = [[ID_acc_dict[k].mean(), ID_acc_dict[k].std()] for k in layer_list]
    y2 = [[nonID_acc_dict[k].mean(), nonID_acc_dict[k].std()] for k in layer_list]
    # y3 = [[SIMI_acc_dict[k].mean(), SIMI_acc_dict[k].std()] for k in layer_list]
    # y4 = [[overlap_acc_dict[k].mean(), overlap_acc_dict[k].std()] for k in layer_list]

    plt.figure()
    plt.plot(x, [item[0] for item in y], 'b', label='full')
    plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y], [min(item[0] + item[1], 1) for item in y], alpha=0.4, edgecolor='k')
    # plt.plot(x, [item[0] for item in y3], 'b', label='SIMI')
    # plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y3], [min(item[0] + item[1], 1) for item in y3], alpha=0.4, edgecolor='k')
    plt.plot(x, [item[0] for item in y1], 'r', label='ID')
    plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y1], [min(item[0] + item[1], 1) for item in y1], alpha=0.4, edgecolor='k')
    plt.plot(x, [item[0] for item in y2], 'gray', label='nonID')
    plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y2], [min(item[0] + item[1], 1) for item in y2], alpha=0.4, edgecolor='k', color='gray')
    # plt.plot(x, [item[0] for item in y4], 'gray', label='overlap')
    # plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y4], [min(item[0] + item[1], 1) for item in y4], alpha=0.4, edgecolor='k', color='gray')

    plt.ylim((0, 1))
    if dataSet_name == 'localizer_balanced':
        plt.legend(loc=3)
    else:
        plt.legend(loc=2)
    plt.xticks(rotation=45)
    plt.ylabel('Avg accuracy')
    plt.title('Classification Accuracy of ' + dataSet_name + ' with error bar')
    plt.savefig(root + '/' + 'OUT/' + dataSet_name + '.png', bbox_inches='tight', dpi=100)
    plt.savefig(root + '/' + 'OUT/' + dataSet_name + '.eps', bbox_inches='tight', dpi=100)
    plt.savefig(root + '/' + 'OUT/' + dataSet_name + '.svg', bbox_inches='tight', dpi=100)
