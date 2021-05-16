'''Apply the ID/nonID mask on different input
    and plot acc result
'''
import os
# import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataSet_name = 'CelebA_original_randDrop_0.3_conv5_2'
root = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_' + dataSet_name
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original_singleDrop_0.3/' + dataSet_name.split('_')[-2] + '_' + dataSet_name.split('_')[-1]
if not os.path.exists(dest):
    os.makedirs(dest)
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
print('layer_list:', layer_list)
full_acc_dict = {}
ID_acc_dict = {}
nonID_acc_dict = {}

print('Now doing', dataSet_name)
for layer in layer_list:
    # Load full feat mat and do ANOVA to filter out sig neuron
    print('****************************')
    print('Loading feature matrix of layer', layer + ':')
    FM_path = os.path.join(root, layer + '_fullMatrix.pkl')
    f = open(FM_path, 'rb')
    full_matrix = pickle.load(f)
    f.close
    print(full_matrix.shape)
    # print('Loading SNI list of layer', layer + ':')
    # SNI_path = os.path.join(sig_ind_dir, layer + '_sig_neuron_ind.pkl')
    # f = open(SNI_path, 'rb')
    # sig_neuron_ind = pickle.load(f)
    # f.close

    # Create label for feat mat
    print('Generating label...')
    label = []
    for i in range(50):
        label += [i + 1] * 10
    # label = np.array(label)
    # label = np.expand_dims(label, axis=1)

    # Load gender/race label
    # print('Loadomg labels...')
    # label_list = np.loadtxt('/home/sdb1/Jinge/ID_selective/Data/labelG.csv', delimiter=',')[:, 1]
    # label = []
    # for i in label_list:
    #     label += [int(i)] * 10
    # label = np.array(label)
    # label = np.expand_dims(label, axis=1)

    print('Doing classification of layer', layer + ':')
    # SVM for full
    full_train, full_test, label_train, label_test = train_test_split(full_matrix, label, stratify=label, test_size=0.33, random_state=42)
    print('full data load complete, doing SVM...')
    print(full_train.shape, full_test.shape)
    samples = len(label_test)
    clf = svm.SVC()
    clf.fit(full_train, label_train)
    full_predicted = clf.predict(full_test)
    full_correct = 0
    for i in range(samples):
        if full_predicted[i] == label_test[i]:
            full_correct += 1
    full_acc = full_correct / samples
    print('full_Accuracy: %d %%' % (100 * full_acc))
    full_acc_dict.update({layer: full_acc})
    del full_train, full_test, label_train, label_test
    print('full data SVM complete, memory released.')

    # Gnerate a boolean mask to return the rest of cols
    # mask = np.array([(i in sig_neuron_ind) for i in range(full_matrix.shape[1])])
    # ID_matrix = full_matrix[:, mask]
    # nonID_matrix = full_matrix[:, ~mask]
    print('Loading mask...')
    SNI_path = os.path.join(mask_dir, layer + '_sig_neuron_ind.pkl')
    f = open(SNI_path, 'rb')
    maskID = pickle.load(f)
    nonSNI_path = os.path.join(mask_dir, layer + '_non_sig_neuron_ind.pkl')
    f = open(nonSNI_path, 'rb')
    maskNonID = pickle.load(f)

    # SVM for ID
    ID_train, ID_test, label_train, label_test = train_test_split(full_matrix[:, maskID], label, stratify=label, test_size=0.33, random_state=42)
    print('ID data load complete, doing SVM...')
    print(ID_train.shape, ID_test.shape)
    clf1 = svm.SVC()
    clf1.fit(ID_train, label_train)
    ID_predicted = clf1.predict(ID_test)
    ID_correct = 0
    for i in range(samples):
        if ID_predicted[i] == label_test[i]:
            ID_correct += 1
    ID_acc = ID_correct / samples
    print('ID_Accuracy: %d %%' % (100 * ID_acc))
    ID_acc_dict.update({layer: ID_acc})
    del ID_train, ID_test, label_train, label_test
    print('ID data SVM complete, memory released.')

    # SVM for non-ID
    nonID_train, nonID_test, label_train, label_test = train_test_split(full_matrix[:, maskNonID], label, stratify=label, test_size=0.33, random_state=42)
    print('nonID data load complete, doing SVM...')
    print(nonID_train.shape, nonID_test.shape)
    clf2 = svm.SVC()
    clf2.fit(nonID_train, label_train)
    nonID_predicted = clf2.predict(nonID_test)
    nonID_correct = 0
    for i in range(samples):
        if nonID_predicted[i] == label_test[i]:
            nonID_correct += 1
    nonID_acc = nonID_correct / samples
    print('nonID_Accuracy: %d %%' % (100 * nonID_acc))
    nonID_acc_dict.update({layer: nonID_acc})
    del nonID_train, nonID_test, label_train, label_test
    del full_matrix
    print('nonID data SVM complete, memory released.')

full_acc_dict_path = dest + '/full_acc_dict.pkl'
with open(full_acc_dict_path, 'wb') as f:
    pickle.dump(full_acc_dict, f)
ID_acc_dict_path = dest + '/ID_acc_dict.pkl'
with open(ID_acc_dict_path, 'wb') as f:
    pickle.dump(ID_acc_dict, f)
nonID_acc_dict_path = dest + '/nonID_acc_dict.pkl'
with open(nonID_acc_dict_path, 'wb') as f:
    pickle.dump(nonID_acc_dict, f)
print(dataSet_name.split('_')[-2] + '_' + dataSet_name.split('_')[-1], 'Acc saved!')

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
plt.plot(x, y2, 'gray', label='nonID')

plt.ylim((0, 1))
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy of ' + dataSet_name)
plt.savefig(dest + '/singleDrop_' + dataSet_name.split('_')[-2] + '_' + dataSet_name.split('_')[-1] + '.png', bbox_inches='tight', dpi=100)
