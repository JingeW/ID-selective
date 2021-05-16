'''
conv5_3 only
u-test
'''

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Data perp: make the feat_matrix
root = '/home/lab321/Jinge/ID_neuron_selectivity/VGG16_ImgNet_feat_mat'
dest = '/home/lab321/Jinge/ID_neuron_selectivity/Result/Runnan_result'
if not os.path.exists(dest):
    os.makedirs(dest)
target_list = ['28']

ID_list = sorted([root + '/' + f for f in os.listdir(root)])
full_matrix = np.zeros(shape=(500, 100352))
count = -1
for ID in ID_list:
    img_list = sorted([ID + '/' + f for f in os.listdir(ID)])
    for img in img_list:
        print('Now processing...', img)
        count += 1
        feat_list = sorted([img + '/' + target_list[0] + '/' + f for f in os.listdir(img + '/' + target_list[0])])
        merge_feat = np.empty([0])
        for feat in feat_list:
            matrix = np.loadtxt(feat, delimiter=',').flatten()
            merge_feat = np.concatenate((merge_feat, matrix))
        full_matrix[count] = merge_feat
print('Data-prep finished!')

np.savetxt(dest + '/' + 'full_matirix.csv', full_matrix, delimiter=',')
full_matrix = np.loadtxt('/home/lab321/Jinge/ID_neuron_selectivity/Result/VGG16_ImgNet/full_matirix.csv', delimiter=',')


# Runnan result test
runnan_path = '/home/lab321/Jinge/ID_neuron_selectivity/FM_vggface_vgg16_conv5_3.csv'
full_matrix = np.loadtxt(runnan_path, delimiter=',').transpose()


# Select significantly responding neurons by Mann-Whitney U test
print('Neuron selection start...')
alpha = 0.01
sig_ind_total = []
for i in range(50):
    print('Now comparing:', str(i) + 'th class to others...')
    A = full_matrix[i * 10: i * 10 + 10, :]
    B = np.vstack((full_matrix[: i * 10, :], full_matrix[i * 10 + 10:, :]))
    sig_ind = []
    for j in range(len(full_matrix[1])):
        # print(str(j)+'th neuron...')
        Ai = A[:, j]
        temp_p = []
        for k in range(49):
            # print(k)
            Bjk = B[k * 10: k * 10 + 10, j]
            try:
                stat, p = mannwhitneyu(Ai, Bjk, alternative='greater')
            except(Exception):
                p = 1  # 2 samples are identical
            temp_p.append(p)
        if max(temp_p) < alpha:
            # print(str(j)+'th neuron is significantly respond to '+str(i)+'th class')
            sig_ind.append(j)
    sig_ind = np.array(sig_ind)
    sig_ind_total.append(sig_ind)
sig_ind_total = np.array(sig_ind_total)
with open('/home/lab321/Jinge/ID_neuron_selectivity/Result/Runnan_result/sig_ind_total.csv', 'wb') as f:
    for row in sig_ind_total:
        np.savetxt(f, [row], fmt='%d', delimiter=',')
print('Neuron selection finished!')

sig_ind_total = pd.read_csv('/home/lab321/Jinge/ID_neuron_selectivity/Result/VGG16_ImgNet/sig_ind_total.csv', header=None, sep='s+', skip_blank_lines=False)
sig_ind_total = sig_ind_total[0].str.split(',', expand=True)

# Count
print('Counting # of selective neurons for each class...')
count = []
for i in range(len(sig_ind_total)):
    count.append(sig_ind_total.loc[i, :].count())  # df.loc[row,col] to locate target vector. df.count() to count non-NaN cell for target vector
print(count)

X = np.arange(0, 50)
plt.bar(X, sorted(count, reverse=True))
plt.xlabel('Classes(sorted)')
plt.ylabel('Counts')
plt.title('Number of selective neurons(sorted)')
plt.show()
plt.bar(X, count)
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Vgg16_ImgNet Number of selective neurons')
plt.show()
print('Counting finished!')

sig_ind_45 = list(map(int, sig_ind_total.iloc[45, :]))
selective_neuron_45 = full_matrix[:, sig_ind_45]
np.savetxt('./Vgg_Img_net_sn_45.csv', selective_neuron_45, delimiter=',')

print('Computing selective index...')
N_img = 10
N_class = 50
class_ind_list = np.arange(N_class)
selectivity_index_total = []
for pref_class in class_ind_list:  # 50 classes
    print('Comparing ' + str(pref_class) + 'th class to others...')
    i_sig_ind = [x for x in sig_ind_total.loc[pref_class, :] if str(x) != 'nan']
    other_class_list = np.delete(class_ind_list, np.argwhere(class_ind_list == pref_class))
    selectivity_index = []
    for i_ind in i_sig_ind:  # significant neuron
        # print(i_ind)
        if i_ind is None:
            continue
        i_class = full_matrix[pref_class * 10: pref_class * 10 + 10, int(i_ind)]
        Pi = []
        for i in i_class:  # i_class has 10 imgs
            Pij = []
            for j_ind in other_class_list:  # other 15 classes
                cnt = len([j for j in full_matrix[j_ind * 10:j_ind * 10 + 10, int(i_ind)] if i > j])
                Pij.append(cnt / N_img)
            Pi.append(np.prod(Pij))
        selectivity_index.append(sum(Pi) / len(Pi))
    selectivity_index_total.append(selectivity_index)
selectivity_index_total = np.array(selectivity_index_total)
with open('/home/lab321/Jinge/ID_neuron_selectivity/selectivity_index_total.csv', 'wb') as f:
    for row in selectivity_index_total:
        np.savetxt(f, [row], delimiter=',')
# face & non-face selective index
selectivity_index_total = pd.read_csv('/home/lab321/Jinge/ID_neuron_selectivity/selectivity_index_total.csv', header=None)
face_selectivity_index = [x for x in selectivity_index_total.loc[0, :] if str(x) != 'nan']
face_selectivity_index_avg = sum(face_selectivity_index) / len(face_selectivity_index)
