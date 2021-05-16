'''Modify the result
'''
import os
import pickle
import matplotlib.pyplot as plt

# sig_ind_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
# dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_ImageNet_random'
dest = '/home/sdb1/Jinge/ID_selective/Result'
save_path = dest + '/ACC'
if not os.path.exists(save_path):
    os.makedirs(save_path)
path_list = [os.path.join(dest, f) for f in os.listdir(dest) if 'VGG16' in f]
for path in path_list:
    if not os.path.exists(path):
        os.makedirs(path)

    path1 = os.path.join(path, 'full_acc_dict.pkl')
    with open(path1, 'rb') as f:
        full_acc_dict = pickle.load(f)
    path2 = os.path.join(path, 'ID_acc_dict.pkl')
    with open(path2, 'rb') as f:
        ID_acc_dict = pickle.load(f)
    path3 = os.path.join(path, 'nonID_acc_dict.pkl')
    with open(path3, 'rb') as f:
        nonID_acc_dict = pickle.load(f)

    # Plot
    layer_squence = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
        'FC6', 'FC7', 'FC8'
    ]

    x = layer_squence
    y = [full_acc_dict[k] for k in layer_squence]
    y1 = [ID_acc_dict[k] for k in layer_squence]
    y2 = [nonID_acc_dict[k] for k in layer_squence]
    plt.figure()
    plt.plot(x, y, 'b', label='full')
    plt.plot(x, y1, 'r', label='ID')
    plt.plot(x, y2, 'g', label='nonID')
    plt.ylim((0, 1))
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    token = path.split('/')[-1].split('_')
    if len(token) == 4:
        title = token[2] + '_' + token[3]
    else:
        title = token[2] + '_' + token[3] + '_' + token[4]
    plt.title('Decoding Accuracy of ' + title)
    plt.savefig(save_path + '/' + 'acc_' + title + '.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + 'acc_' + title + '.eps', format='eps', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + 'acc_' + title + '.svg', format='svg', bbox_inches='tight', dpi=100)
