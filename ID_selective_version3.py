'''
All layers
'''

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# # Data prep
# root = '/home/sdb1/Jinge/ID_selective/VGG16_ImgNet_feat_mat_full'
# dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_ImgNet_full'
# if not os.path.exists(dest):
#     os.makedirs(dest)
# target_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_ImgNet_feat_mat_full/01/002475'))
# sample_num = 50*10

# for layer in target_list:
#     print('Building feature matrix of layer',layer+':')
#     ID_list = sorted([root+'/'+f for f in os.listdir(root)])
#     full_matrix = np.empty([0])
#     for ID in ID_list:
#         img_list = sorted([ID+'/'+f for f in os.listdir(ID)])
#         for img in img_list:
#             print('Now processing...', img)
#             feat_list = sorted([img+'/'+layer+'/'+f for f in os.listdir(img+'/'+layer)])
#             merge_feat = np.empty([0])
#             for feat in feat_list:
#                 matrix = np.loadtxt(feat,delimiter=',').flatten()
#                 merge_feat = np.concatenate((merge_feat,matrix))
#             full_matrix = np.concatenate((full_matrix,merge_feat))
#     full_matrix = full_matrix.reshape((sample_num,-1))
#     print(full_matrix.shape)
#     np.savetxt(dest+'/'+layer+'_full_matirix.csv',full_matrix,delimiter=',')
# print('Data-prep finished!')


# Stats test
def make_layer_name(string):
    name = string.split('/')[-1].split('.')[0].split('_')
    check_list = ['1', '2', '3']
    if name[2] in check_list:
        layer_name = name[0] + '_' + name[1] + '_' + name[2]
    else:
        layer_name = name[0] + '_' + name[1]
    return layer_name


def make_fullMatrix(path):
    name = make_layer_name(path)
    # print('Building feature matrix of layer',name+':')
    full_matrix = np.loadtxt(path, delimiter=',')
    # print('shape of full_matrix:', full_matrix.shape)
    return name, full_matrix


def oneWay_ANOVA(layer_name, matrix, num_per_class, dest, alpha):
    row, col = matrix.shape
    pl = []
    for i in range(col):
        neuron = matrix[:, i]
        d = [neuron[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        pl.append(p)
    pl = np.array(pl)
    np.savetxt(dest + '/' + layer_name + '_plist.csv', pl, delimiter=',')
    sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
    print(layer_name, 'has', len(sig_neuron_ind), ' significant neurons')
    percent = len(sig_neuron_ind) / col * 100
    print('Percentage: {:.2f}%'.format(percent))
    return sig_neuron_ind, percent


def main():
    full_matrix_root = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_full'
    stats_dest = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
    if not os.path.exists(stats_dest):
        os.makedirs(stats_dest)
    num_per_class = 10
    alpha = 0.01
    full_matrix_path_list = sorted([full_matrix_root + '/' + f for f in os.listdir(full_matrix_root)])
    full_matrix_path_list = full_matrix_path_list[10:]
    count = {}
    cnt = []
    percent_list = []
    for full_matrix_path in full_matrix_path_list:
        layer_name, full_matrix = make_fullMatrix(full_matrix_path)
        sig_neuron_ind, percent = oneWay_ANOVA(layer_name, full_matrix, num_per_class, stats_dest, alpha)
        num = len(sig_neuron_ind)
        count.update({layer_name: num})
        cnt.append(num)
        percent_list.append(percent)
    cnt = np.array(cnt)
    percent_list = np.array(percent_list)
    np.savetxt(stats_dest + '/' + 'count.csv', cnt, delimiter=',')
    np.savetxt(stats_dest + '/' + 'percent.csv', percent_list, delimiter=',')

    # plot
    x = count.keys()
    y = count.values()
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts')
    plt.title('Significant neuron counts for different layers')
    plt.savefig(stats_dest + '/' + 'count.png')
    plt.show()

    x = count.keys()
    y = percent_list
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts percentage')
    plt.title('Significant neuron counts percentage for different layers')
    plt.savefig(stats_dest + '/' + 'percent.png')
    plt.show()


if __name__ == "__main__":
    main()
