'''
Check mean and std in each neuron
'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random


# ==============plot local means for different IDs among a sig_neuron===============

root = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
file_path_list = sorted([os.path.join(root, f) for f in os.listdir(root) if 'neuron' in f.split('_')[-1]])
# file_path = random.choice(file_path_list)
save_path = root + '/' + 'Mean_check'
if not os.path.exists(save_path):
    os.makedirs(save_path)

fig, axs = plt.subplots(7, 3, figsize=((20, 20)))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
x = np.arange(1, 51)
cnt_row = 0
cnt_col = 0

for file_path in file_path_list:
    layer = file_path.split('/')[-1].split('.')[0]
    num_per_class = 10

    f = open(file_path, 'rb')
    sig_matrix = pickle.load(f)
    f.close()
    row, col = sig_matrix.shape
    check_neuron = random.choice(range(col))
    print('Now check', layer, ': #', check_neuron, '\n')
    neuron_vector = sig_matrix[:, check_neuron]
    ID_vector_list = [neuron_vector[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
    ID_vector_list = np.array(ID_vector_list)
    mean_list = np.mean(ID_vector_list, axis=1)
    std_list = np.std(ID_vector_list, axis=1)
    se_list = std_list / np.sqrt(num_per_class)
    x = np.arange(50) + 1

    # plt.figure()
    # plt.bar(x, mean_list, yerr=se_list)
    # plt.ylabel('local mean of each ID')
    # plt.title(layer + ' # ' + str(check_neuron))
    # plt.tight_layout()

    # plt.savefig(save_path + '/' + layer + '_' + str(check_neuron) + '.png', bbox_inches='tight', dpi=100)
    # plt.savefig(save_path + '/' + layer + '_' + str(check_neuron) + '.eps', bbox_inches='tight', dpi=100, format='eps')
    # plt.savefig(save_path + '/' + layer + '_' + str(check_neuron) + '.svg', bbox_inches='tight', dpi=100, format='svg')

    axs[cnt_row, cnt_col].bar(x, mean_list, width=0.5)
    axs[cnt_row, cnt_col].set_title(layer + ' # ' + str(check_neuron), fontsize=14)
    cnt_col += 1
    if cnt_col > 2:
        cnt_col = 0
        cnt_row += 1
for ax in axs.flat:
    ax.label_outer()
for ax in axs.flat:
    ax.set(xlabel='IDs', ylabel='local mean')
plt.savefig(save_path + '/' + 'meanCheckAll.png', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'meanCheckAll.eps', format='eps', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'meanCheckAll.svg', format='svg', bbox_inches='tight', dpi=100)
