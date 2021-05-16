'''Do SVM on SIMI masked neuron for each layer
'''
import os
import pickle
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt


def makeFolder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def makeLabels(sample_num, class_num):
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label


def SVM_classification_KFlod(matrix, label, k):
    clf = svm.SVC()
    CV = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # CV = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=42)
    acc = cross_val_score(clf, matrix, label, cv=CV)
    return acc


# *********Adjustable***********
dataSet_name = 'CelebA_cartoon_Hosoda'
k = 5
sample_num = 10
class_num = 50
# ******************************

print('=========', str(k) + '-Folder SVM for', dataSet_name, 'SIMI', '=========')
FM_dir = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_' + dataSet_name
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
save_dir = '/home/sdb1/Jinge/ID_selective/kFolder/' + dataSet_name
makeFolder(save_dir)
label = makeLabels(sample_num, class_num)
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]

SIMI_acc_dict = {}
# nonSIMI_acc_dict = {}
for layer in layer_list:
    print('****************************')
    print('Loading feature matrix of layer', layer + ':')
    FM_path = os.path.join(FM_dir, layer + '_fullMatrix.pkl')
    with open(FM_path, 'rb') as f:
        fullMatrix = pickle.load(f)
    print(fullMatrix.shape)

    print('Loading mask...')
    with open(mask_dir + '/' + layer + '_SIMI_ind.pkl', 'rb') as f:
        SIMI = pickle.load(f)
    # nonSIMI = list(set(range(fullMatrix.shape[1])) - set(SIMI))
    # print('Length of SIMI/nonSIMI mask:', len(SIMI), len(nonSIMI))

    print('Doing SVM for SIMI...')
    SIMI_acc = SVM_classification_KFlod(fullMatrix[:, SIMI], label, k)
    SIMI_acc_dict.update({layer: SIMI_acc})
    print(SIMI_acc)
    print("Accuracy: %0.2f (+/- %0.2f)" % (SIMI_acc.mean(), SIMI_acc.std() * 2))

    # print('Doing SVM for nonSIMI...')
    # nonSIMI_acc = SVM_classification_KFlod(fullMatrix[:, nonSIMI], label, k)
    # nonSIMI_acc_dict.update({layer: nonSIMI_acc})
    # print('ID_Accuracy: %d %%' % (100 * nonSIMI_acc))

with open(save_dir + '/SIMI_acc_dict.pkl', 'wb') as f:
    pickle.dump(SIMI_acc_dict, f)
# with open(save_dir + '/nonSIMI_acc_dict.pkl', 'wb') as f:
#     pickle.dump(nonSIMI_acc_dict, f)
print('ACC saved!')

# plot
x = layer_list
y = [[SIMI_acc_dict[k].mean(), SIMI_acc_dict[k].std()] for k in layer_list]

# bar plot for avg and std
plt.figure()
plt.bar(x, [item[0] for item in y], yerr=[item[1] for item in y], width=0.5)
plt.xticks(rotation=45)
plt.ylim((0, 1))
plt.ylabel('Avg acuracy')
plt.title('Average acuracy of SIMI')
plt.savefig(save_dir + '/bar_SIMI.png', bbox_inches='tight', dpi=100)

# avg acc with error bar
plt.figure()
plt.plot(x, [item[0] for item in y], 'b', label='full')
plt.fill_between(x,  [max(item[0] - item[1], 0) for item in y], [min(item[0] + item[1], 1) for item in y], alpha=0.4, edgecolor='b')
plt.ylim((0, 1))
if dataSet_name == 'localizer_balanced':
    plt.legend(loc=3)
else:
    plt.legend(loc=2)
plt.xticks(rotation=45)
plt.ylabel('Avg accuracy')
plt.title('Classification Accuracy of ' + dataSet_name + ' with error bar')
plt.savefig(save_dir + '/acc_SIMI.png', bbox_inches='tight', dpi=100)
