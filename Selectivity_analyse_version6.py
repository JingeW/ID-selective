'''Stack plot for SI/MI in each layer
'''
import os
import pickle
import matplotlib.pyplot as plt

path = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full/SIMI_cnt.pkl'
f = open(path, 'rb')
SIMI_dict = pickle.load(f)
f.close()

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
    'FC6': 4096, 'FC7': 4096, 'FC8': 2622,
}

save_path = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full/SIMI'
if not os.path.exists(save_path):
    os.makedirs(save_path)

x = layer_squence
y_list = [SIMI_dict[k] for k in layer_squence]
# print(y_list)
y1 = [item[0] for item in y_list]
y2 = [item[1] for item in y_list]
y = [i + j for i, j in zip(y1, y2)]
t = [total_neuron[k] for k in layer_squence]
percent_si = [i / j * 100 for i, j in zip(y1, t)]
percent_mi = [i / j * 100 for i, j in zip(y2, t)]
# print(y1)
# print(y2)
plt.figure(1)
p1 = plt.bar(x, y1, width=0.5)
p2 = plt.bar(x, y2, width=0.5)
plt.ylabel('Num of neurons')
plt.xticks(rotation=45)
plt.legend((p1[0], p2[0]), ('SI', 'MI'))
plt.title('Stack plot for SI/MI num in each layer')
plt.savefig(save_path + '/stackplt_num.png', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/stackplt_num.eps', bbox_inches='tight', dpi=100, format='eps')
plt.savefig(save_path + '/stackplt_num.svg', bbox_inches='tight', dpi=100, format='svg')
plt.figure(2)
p3 = plt.bar(x, percent_si, width=0.5)
p4 = plt.bar(x, percent_mi, bottom=percent_si, width=0.5)
plt.ylim((0, 100))
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Neuron', 'multiple_Identity(MI) Neuron'), frameon=False)
plt.title('Stack plot for SI/MI percentage in each layer')
plt.savefig(save_path + '/stackplt_percentage.png', bbox_inches='tight', dpi=100)
plt.savefig(save_path + '/stackplt_percentage.eps', bbox_inches='tight', dpi=100, format='eps')
plt.savefig(save_path + '/stackplt_percentage.svg', bbox_inches='tight', dpi=100, format='svg')

# # Plot for Decoding Accuracy subset
# layer_squence_sub = [
#     'Pool1',
#     'Pool2',
#     'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
#     'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
#     'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
#     'FC6', 'FC7', 'FC8'
# ]

# full_acc = {
#     'Pool5': 0.75, 'Pool4': 0.15, 'Pool3': 0.09, 'Pool2': 0.04, 'Pool1': 0.03,
#     'FC8': 0.81, 'FC7': 0.84, 'FC6': 0.86,
#     'Conv5_3': 0.73, 'Conv5_2': 0.37, 'Conv5_1': 0.21,
#     'Conv4_3': 0.12, 'Conv4_2': 0.09, 'Conv4_1': 0.08,
#     'Conv3_3': 0.07, 'Conv3_2': 0.07, 'Conv3_1': 0.07,
# }

# ID_acc = {
#     'Pool5': 0.78, 'Pool4': 0.35, 'Pool3': 0.30, 'Pool2': 0.09, 'Pool1': 0.04,
#     'FC8': 0.81, 'FC7': 0.84, 'FC6': 0.86,
#     'Conv5_3': 0.78, 'Conv5_2': 0.52, 'Conv5_1': 0.39,
#     'Conv4_3': 0.36, 'Conv4_2': 0.30, 'Conv4_1': 0.29,
#     'Conv3_3': 0.32, 'Conv3_2': 0.26, 'Conv3_1': 0.16,
# }

# nonID_acc = {
#     'Pool5': 0.01, 'Pool4': 0.00, 'Pool3': 0.00, 'Pool2': 0.00, 'Pool1': 0.00,
#     'FC8': 0.01, 'FC7': 0.02, 'FC6': 0.02,
#     'Conv5_3': 0.01, 'Conv5_2': 0.00, 'Conv5_1': 0.00,
#     'Conv4_3': 0.00, 'Conv4_2': 0.00, 'Conv4_1': 0.00,
#     'Conv3_3': 0.00, 'Conv3_2': 0.00, 'Conv3_1': 0.00,
# }


# plt.figure(3)
# x = layer_squence_sub
# y = [full_acc[k] for k in layer_squence_sub]
# y1 = [ID_acc[k] for k in layer_squence_sub]
# y2 = [nonID_acc[k] for k in layer_squence_sub]
# plt.figure()
# plt.plot(x, y, 'b', label='full')
# plt.plot(x, y1, 'r', label='ID')
# plt.plot(x, y2, 'g', label='nonID')
# plt.legend()
# plt.xticks(rotation=45)
# plt.ylabel('Classification Accuracy')
# plt.title('Decoding Accuracy')
# plt.savefig('./acc_sub.png')
