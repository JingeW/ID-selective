''' Using python 'path'.py > 'path'.txt
    to record the print() content into a txt file
'''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# instruction = '''
# Data: CelebA 50IDs, 10 imgs for each. 500 imgs intotal.
# Feature: extract from vgg16 vggface pretrained
#          Full feature map, the dim of each layer equal to the 500*flatten(channel*w*h)

# THRESHOLD = global_mean + 2*global_std

# For each neuron in a certain layer:
#     global_mean is the mean of the feature values of all 500 imgs
#     global_std is the std of all 500 values above
#     local_mean is the mean within a single ID, which has 10 imgs
#     So, we have 50 local means to compare with the THRESHOLD
#     Let's say "Neuron encode a class, if the local mean larger than the THRESHOLD"
# '''
# print(instruction)


def layer_name(string):
    name = string.split('/')[-1].split('.')[0].split('_')
    check_list = ['1', '2', '3']
    if name[1] in check_list:
        layer = name[0] + '_' + name[1]
    else:
        layer = name[0]
    return layer


sig_neuron_dir = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
num_per_class = 10
pkl_list = sorted([os.path.join(sig_neuron_dir, f) for f in os.listdir(sig_neuron_dir) if 'neuron' in f.split('_')[-1]])

# SIMI_dict = {}
# for pkl_file in pkl_list:
#     layer = layer_name(pkl_file)
#     path = os.path.join(sig_neuron_dir, pkl_file)
#     with open(path, 'rb') as temp:
#         sig_neuron = pickle.load(temp)
#         row, col = sig_neuron.shape
#         # print(sig_neuron.shape)
#         print(layer + ':', col, 'selective neuron intotal')
#         cnt = 0
#         cnt_si = 0
#         cnt_mi = 0
#         for i in range(col):
#             neuron = sig_neuron[:, i]
#             global_mean = np.mean(neuron)
#             global_std = np.std(neuron)
#             threshold = global_mean + 2 * global_std
#             d = [neuron[i * num_per_class:i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
#             d = np.array(d)
#             local_mean = np.mean(d, axis=1)
#             encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]
#             if not encode_class == []:
#                 print('\tneuron', i, 'encode class:', encode_class)
#                 cnt += 1
#                 if len(encode_class) == 1:
#                     cnt_si += 1
#                 else:
#                     cnt_mi += 1
#         print(cnt, 'neuron pass the threhold')
#         print('SI:', cnt_si)
#         print('MI:', cnt_mi, '\n')
#         SIMI_dict.update({layer: [cnt_si, cnt_mi]})
# with open(sig_neuron_dir + '/SIMI_cnt.pkl', 'wb') as f:
#     pickle.dump(SIMI_dict, f)

encode_class_dict = {}
for pkl_file in pkl_list:  # loop in layers
    layer = layer_name(pkl_file)
    print('now processing:', layer)
    with open(pkl_file, 'rb') as temp:
        sig_neuron = pickle.load(temp)
        row, col = sig_neuron.shape
        print(sig_neuron.shape)
        # print(layer+':', col, 'selective neuron intotal')
        cnt = 0
        encode_class = []
        for i in range(col):  # loop in neurons
            neuron = sig_neuron[:, i]
            global_mean = np.mean(neuron)
            global_std = np.std(neuron)
            threshold = global_mean + 2 * global_std
            d = [neuron[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
            d = np.array(d)
            local_mean = np.mean(d, axis=1)
            for i, mean in enumerate(local_mean):
                if mean > threshold:
                    encode_class.append(i + 1)
    encode_class_dict.update({layer: encode_class})

layer_squence = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]

total_neuron = {
    'Conv1_1': 224 * 224 * 64, 'Conv1_2': 224 * 224 * 64, 'Pool1': 112 * 112 * 128,
    'Conv2_1': 112 * 112 * 128, 'Conv2_2': 112 * 112 * 128, 'Pool2': 56 * 56 * 256,
    'Conv3_1': 56 * 56 * 256, 'Conv3_2': 56 * 56 * 256, 'Conv3_3': 56 * 56 * 256, 'Pool3': 28 * 28 * 512,
    'Conv4_1': 28 * 28 * 512, 'Conv4_2': 28 * 28 * 512, 'Conv4_3': 28 * 28 * 512, 'Pool4': 14 * 14 * 512,
    'Conv5_1': 14 * 14 * 512, 'Conv5_2': 14 * 14 * 512, 'Conv5_3': 14 * 14 * 512, 'Pool5': 7 * 7 * 512,
    'FC6': 4096, 'FC7': 4096, 'FC8':  2622,
}

save_path = sig_neuron_dir + '/Freq'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(sig_neuron_dir, 'encode_class_dict.pkl'), 'wb') as f:
    pickle.dump(encode_class_dict, f)

freq_dic = {}
for layer in layer_squence:
    freq = {}
    encode_class_list = encode_class_dict[layer]
    for item in encode_class_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    freq = {k: v / total_neuron[layer] for k, v in freq.items()}
    freq = dict(sorted(freq.items(), key=lambda item: item[0]))
    freq_dic.update({layer: freq})
a = pd.DataFrame.from_dict(freq_dic)

plt.figure()
im = plt.matshow(a, aspect='auto')
plt.colorbar(im, fraction=0.12, pad=0.04)
plt.xlabel('Layers')
plt.ylabel('IDs')
plt.title('Ecode Frequency for Each Layer')
plt.savefig(save_path + '/' + 'Ecode_Frequency_for_Each_Layer.png', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'Ecode_Frequency_for_Each_Layer.eps', format='eps', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'Ecode_Frequency_for_Each_Layer.svg', format='svg', bbox_inches='tight', dpi=100)

occ_list = []
for layer in layer_squence:
    occurrences = []
    for i in range(50):
        occ = encode_class_dict[layer].count(i + 1)
        occurrences.append(occ)
    occ_list.append(occurrences)
    x = np.arange(1, 51)
    plt.figure()
    plt.bar(x, occurrences, width=0.5)
    plt.xticks(np.arange(0, 51, step=2))
    plt.xlabel('IDs')
    plt.ylabel('Frequrency')
    plt.title('Encoded ID frequency: ' + layer + '\nTh: 2std')
    plt.savefig(save_path + '/' + layer + '_sigFreq.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + layer + '_sigFreq.eps', format='eps', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/' + layer + '_sigFreq.svg', format='svg', bbox_inches='tight', dpi=100)

    # plt.show()
# np.array(occ_list)
# np.savetxt(sig_neuron_dir + '/occ_list.txt', occ_list, fmt='%d')

fig, axs = plt.subplots(7, 3, figsize=((20, 20)))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
x = np.arange(1, 51)
cnt_row = 0
cnt_col = 0
for layer in layer_squence:
    occurrences = []
    for i in range(50):
        occ = encode_class_dict[layer].count(i + 1)
        occurrences.append(occ)
    axs[cnt_row, cnt_col].bar(x, occurrences, width=0.5)
    axs[cnt_row, cnt_col].set_title(layer, fontsize=14)
    cnt_col += 1
    if cnt_col > 2:
        cnt_col = 0
        cnt_row += 1
for ax in axs.flat:
    ax.label_outer()
for ax in axs.flat:
    ax.set(xlabel='IDs', ylabel='Freq')
plt.savefig(save_path + '/' + 'sigFreqAll.png', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'sigFreqAll.eps', format='eps', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/' + 'sigFreqAll.svg', format='svg', bbox_inches='tight', dpi=100)
