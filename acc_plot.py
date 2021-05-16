'''Plot acc including overlap
'''
import os
import pickle
import matplotlib.pyplot as plt

# path1 = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original/ID_acc_dict.pkl'
# path2 = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_random/ID_acc_dict.pkl'
# overlap_path = '/home/sdb1/Jinge/ID_selective/Comparison/CelebA_original_vs_CelebA_random/overlap_acc_dict.pkl'
path = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original_race'

path1 = os.path.join(path, 'full_acc_dict.pkl')
with open(path1, 'rb') as f:
    full_acc_dict = pickle.load(f)
path2 = os.path.join(path, 'ID_acc_dict.pkl')
with open(path2, 'rb') as f:
    ID_acc_dict = pickle.load(f)
path3 = os.path.join(path, 'nonID_acc_dict.pkl')
with open(path3, 'rb') as f:
    nonID_acc_dict = pickle.load(f)
# path4 = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full/SIMI_acc_dict.pkl'
# with open(path4, 'rb') as f:
#     SIMI_acc_dict = pickle.load(f)
# path5 = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original/nonSIMI_acc_dict.pkl'
# with open(path5, 'rb') as f:
#     nonSIMI_acc_dict = pickle.load(f)
# path6 = '/home/sdb1/Jinge/ID_selective/Comparison/CelebA_original_vs_CelebA_random/overlap_acc_dict.pkl'
# with open(path6, 'rb') as f:
#     overlap_acc_dict = pickle.load(f)
# with open(overlap_path, 'rb') as f:
#     overlap_acc_dict = pickle.load(f)
# with open(path1, 'rb') as f:
#     ID_acc_dict1 = pickle.load(f)
# with open(path2, 'rb') as f:
#     ID_acc_dict2 = pickle.load(f)

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
# y3 = [SIMI_acc_dict[k] for k in layer_squence]
# y = [ID_acc_dict1[k] for k in layer_squence]
# y1 = [ID_acc_dict2[k] for k in layer_squence]
# y4 = [nonSIMI_acc_dict[k] for k in layer_squence]
# y5 = [overlap_acc_dict[k] for k in layer_squence]

plt.figure()
plt.plot(x, y, 'b', label='full')
plt.plot(x, y1, 'r', label='ID')
plt.plot(x, y2, 'k', label='nonID')
# plt.plot(x, y3, 'b', label='SIMI')
# plt.plot(x, y, 'b', label='ID_ori')
# plt.plot(x, y1, 'r', label='ID_rand')
# plt.plot(x, y4, 'b--', label='nonSIMI')
# plt.plot(x, y5, 'g', label='overlap')
plt.ylim((0, 1))
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Classification Accuracy')
title = path.split('/')[-1]
# title = 'Original and Random overlap'
# title = 'SingleLayerDrop_30%_' + path.split('/')[-1]
plt.title('Classification Accuracy of ' + title)
plt.savefig(path + '/' + 'acc_' + title + '.png', bbox_inches='tight', dpi=100)
# plt.savefig(path + '/' + 'acc_withSIMI_' + title + '.png', bbox_inches='tight', dpi=100)
plt.savefig(path + '/' + 'acc_' + title + '.eps', bbox_inches='tight', dpi=100)
plt.savefig(path + '/' + 'acc_' + title + '.svg', bbox_inches='tight', dpi=100)
