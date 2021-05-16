import os
import pickle
import datetime
import numpy as np
from thundersvm import SVC
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from sklearn.model_selection import StratifiedKFold, cross_val_score
starttime = datetime.datetime.now()


def makeFolder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def makeLabels(sample_num, class_num):
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label


def SVM_classification_KFlod(matrix, label, k):
    clf = SVC(gpu_id=2)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc = []
    for train_index, test_index in skf.split(matrix, label):
        matrix_train, matrix_test = matrix[train_index], matrix[test_index]
        label_train, label_test = label[train_index], label[test_index]
        clf.fit(matrix_train, label_train)
        accuracy = clf.score(matrix_test, label_test)
        acc.append(accuracy)
    return acc

# def SVM_classification_KFlod(matrix, label, k):
#     clf = SVC(gpu_id=2)
#     CV = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#     # CV = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=42)
#     acc = cross_val_score(clf, matrix, label, cv=CV)
#     return acc


# *********Adjustable***********
dataSet_name = 'CelebA_original'
k = 10
sample_num = 10
class_num = 50
# ******************************

print('=========', str(k) + '-Folder SVM for', dataSet_name, '=========')
FM_dir = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_' + dataSet_name
# result_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_' + dataSet_name
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
save_dir = '/home/sdb1/Jinge/ID_selective/kFolder_GPU/' + dataSet_name
makeFolder(save_dir)
label = makeLabels(sample_num, class_num)
layer_list = [
    # 'Conv1_1', 'Conv1_2', 'Pool1',
    # 'Conv2_1', 'Conv2_2', 'Pool2',
    # 'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    # 'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
full_acc_dict = {}
ID_acc_dict = {}
nonID_acc_dict = {}

for layer in layer_list:
    print('****************************')
    print('Now working on layer:', layer)

    print('Loading feature matrix...')
    with open(FM_dir + '/' + layer + '_fullMatrix.pkl', 'rb') as f:
        fullMatrix = pickle.load(f)

    print('Doing SVM for full...')
    acc_full = SVM_classification_KFlod(fullMatrix, label, k)
    print(acc_full)
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc_full.mean(), acc_full.std() * 2))
    full_acc_dict.update({layer: acc_full})

    print('Loading mask...')
    with open(mask_dir + '/' + layer + '_sig_neuron_ind.pkl', 'rb') as f:
        maskID = pickle.load(f)
    with open(mask_dir + '/' + layer + '_non_sig_neuron_ind.pkl', 'rb') as f:
        maskNonID = pickle.load(f)
    print('Length of ID/nonID mask:', len(maskID), len(maskNonID))

    print('Doing SVM for ID...')
    acc_ID = SVM_classification_KFlod(fullMatrix[:, maskID], label, k)
    print(acc_ID)
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc_ID.mean(), acc_ID.std() * 2))
    ID_acc_dict.update({layer: acc_ID})

    print('Doing SVM for nonID...')
    acc_nonID = SVM_classification_KFlod(fullMatrix[:, maskNonID], label, k)
    print(acc_nonID)
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc_nonID.mean(), acc_nonID.std() * 2))
    nonID_acc_dict.update({layer: acc_nonID})
    del fullMatrix

with open(save_dir + '/full_acc_dict.pkl', 'wb') as f:
    pickle.dump(full_acc_dict, f)
with open(save_dir + '/ID_acc_dict.pkl', 'wb') as f:
    pickle.dump(ID_acc_dict, f)
with open(save_dir + '/nonID_acc_dict.pkl', 'wb') as f:
    pickle.dump(nonID_acc_dict, f)


# plot
x = layer_list
y = [[full_acc_dict[k].mean(), full_acc_dict[k].std()] for k in layer_list]
y1 = [[ID_acc_dict[k].mean(), ID_acc_dict[k].std()] for k in layer_list]
y2 = [[nonID_acc_dict[k].mean(), nonID_acc_dict[k].std()] for k in layer_list]

# bar plot for avg and std
plt.figure()
plt.bar(x, [item[0] for item in y], yerr=[item[1] for item in y], width=0.5)
plt.xticks(rotation=45)
plt.ylim((0, 1))
plt.ylabel('Avg acuracy')
plt.title('Average acuracy of full')
plt.savefig(save_dir + '/bar_full.png', bbox_inches='tight', dpi=100)

plt.figure()
plt.bar(x, [item[0] for item in y1], yerr=[item[1] for item in y1], width=0.5)
plt.xticks(rotation=45)
plt.ylim((0, 1))
plt.ylabel('Avg acuracy')
plt.title('Average acuracy of ID')
plt.savefig(save_dir + '/bar_ID.png', bbox_inches='tight', dpi=100)

plt.figure()
plt.bar(x, [item[0] for item in y2], yerr=[item[1] for item in y2], width=0.5)
plt.xticks(rotation=45)
plt.ylim((0, 1))
plt.ylabel('Avg acuracy')
plt.title('Average acuracy of nonID')
plt.savefig(save_dir + '/bar_nonID.png', bbox_inches='tight', dpi=100)

# avg acc plot
plt.figure()
plt.plot(x, [item[0] for item in y], 'b', label='full')
plt.plot(x, [item[0] for item in y1], 'r', label='ID')
plt.plot(x, [item[0] for item in y2], 'g', label='nonID')
plt.ylim((0, 1))
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Avg accuracy')
plt.title('Classification Accuracy of ' + dataSet_name)
plt.savefig(save_dir + '/acc.png', bbox_inches='tight', dpi=100)

# avg acc with error bar
fig, ax = plt.subplots()
trans1 = Affine2D().translate(-0.05, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.05, 0.0) + ax.transData
ax.errorbar(x, [item[0] for item in y], yerr=[item[1] for item in y], color='b', label='full', transform=trans1)
ax.errorbar(x, [item[0] for item in y1], yerr=[item[1] for item in y1], color='r', label='ID', transform=trans2)
ax.errorbar(x, [item[0] for item in y2], yerr=[item[1] for item in y2], color='g', label='nonID')
plt.ylim((0, 1))
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Avg accuracy')
plt.title('Classification Accuracy of ' + dataSet_name + ' with error bar')
plt.savefig(save_dir + '/acc_error.png', bbox_inches='tight', dpi=100)

endtime = datetime.datetime.now()
print('Processing time:', endtime - starttime)
