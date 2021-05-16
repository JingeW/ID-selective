'''
All conv layers
ANOVA
'''
import os
import numpy as np
# import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data prep
root = '/home/lab321/Jinge/ID_neuron_selectivity/VGG16_ImgNet_feat_mat'
dest = '/home/lab321/Jinge/ID_neuron_selectivity/Result/VGG16_ImgNet'
if not os.path.exists(dest):
    os.makedirs(dest)
target_list = ['0', '2', '5', '7', '10', '12', '14', '17', '19', '21', '24', '26', '28']
sample_num = 50 * 10

for layer in target_list:
    print('Building feature matrix of layer', layer + ':')
    ID_list = sorted([root + '/' + f for f in os.listdir(root)])
    full_matrix = np.empty([0])
    # count = -1
    for ID in ID_list:
        img_list = sorted([ID + '/' + f for f in os.listdir(ID)])
        for img in img_list:
            print('Now processing...', img)
            # count + = 1
            feat_list = sorted([img + '/' + layer + '/' + f for f in os.listdir(img + '/' + layer)])
            merge_feat = np.empty([0])
            for feat in feat_list:
                matrix = np.loadtxt(feat, delimiter=',').flatten()
                merge_feat = np.concatenate((merge_feat, matrix))
            full_matrix = np.concatenate((full_matrix, merge_feat))
    full_matrix = full_matrix.reshape((sample_num, -1))
    print(full_matrix.shape)
    np.savetxt(dest + '/' + 'layer' + layer + '_full_matirix.csv', full_matrix, delimiter=', ')
print('Data-prep finished!')

# Stats test
full_matrix_root = '/home/lab321/Jinge/ID_neuron_selectivity/Result/VGG16_ImgNet'
stats_dest = '/home/lab321/Jinge/ID_neuron_selectivity/Stats/VGG16_ImgNet'
if not os.path.exists(stats_dest):
    os.makedirs(stats_dest)
full_matrix_path_list = [full_matrix_root + '/' + f for f in os.listdir(full_matrix_root)]
count = []
for full_matrix_path in sorted(full_matrix_path_list, reverse=True):
    layer_name = full_matrix_path.split('/')[-1].split('_')[0]
    print('ANOVA for', layer_name + ':')
    full_matrix = np.loadtxt(full_matrix_path, delimiter=',')
    print(layer_name, 'feature matrix loading cmplete.')
    alpha = 0.01
    pl = []
    # ANOVA for each neuron
    for i in range(len(full_matrix[1])):
        # print('Now processing', str(i + 1) + 'th neuron...')
        neuron = full_matrix[:, i]
        d = [neuron[i * 10: i * 10 + 10] for i in range(50)]
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        pl.append(p)
    pl = np.array(pl)
    np.savetxt(stats_dest + '/' + layer_name + '_plist.csv', pl, delimiter=',')
    sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
    print(layer_name, 'has', len(sig_neuron_ind), 'significant neurons')
    count.append(len(sig_neuron_ind))
count = np.array(count)
np.savetxt(stats_dest + '/count.csv', count, delimiter=',')
print('Count finished!')

# plot
plist_path = [stats_dest + '/' + f for f in os.listdir(stats_dest)]
alpha = 0.01
count = {}
for plist in sorted(plist_path):
    key = plist.split('/')[-1].split('_')[0]
    pl = np.loadtxt(plist, delimiter=',')
    sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
    value = len(sig_neuron_ind)
    count.update({key: value})
print(count)

layers = ['conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
plt.bar(layers, count.values(), width=0.5)
plt.xticks(rotation=45)
plt.ylabel('sig neuron counts')
plt.title('Significant neuron counts for different conv layers')
plt.show()

dim_conv2 = 112 * 112 * 128
dim_conv3 = 56 * 56 * 256
dim_conv4 = 28 * 28 * 512
dim_conv5 = 14 * 14 * 512
dim_list = [dim_conv2, dim_conv2, dim_conv3, dim_conv3, dim_conv3, dim_conv4, dim_conv4, dim_conv4, dim_conv5, dim_conv5, dim_conv5]
ratio = [round(a / b * 100) for a, b in zip(list(count.values()), dim_list)]
plt.bar(layers, ratio, width=0.5)
plt.xticks(rotation=45)
plt.ylabel('percentage')
plt.title('selective neuron ratio for each conv layer')
plt.show()
