'''SIMI decoding acc
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


def SVM_classification(matrix, label):
    matrix_train, matrix_test, label_train, label_test = train_test_split(matrix, label, test_size=0.33, random_state=42)
    print('Shape of train/test:', matrix_train.shape, matrix_test.shape)
    clf = svm.SVC()
    clf.fit(matrix_train, label_train)
    predicted = clf.predict(matrix_test)
    correct = 0
    samples = len(label_test)
    for i in range(samples):
        if predicted[i] == label_test[i]:
            correct += 1
    acc = correct / samples
    return acc


def makeLabels(sample_num, class_num):
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label


sig_neuron_dir = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
sample_num = 10
class_num = 50
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
for layer in layer_list:
    print('Now processing layer:', layer)
    print('Loading sig_neuron...')
    sig_neuron_path = sig_neuron_dir + '/' + layer + '_sig_neuron.pkl'
    with open(sig_neuron_path, 'rb') as f:
        sig_neuron = pickle.load(f)

    print('Generating SIMI...')
    col = sig_neuron.shape[1]
    SIMI_ind = []
    SI_ind = []
    MI_ind = []
    for i in range(col):
        neuron = sig_neuron[:, i]
        global_mean = np.mean(neuron)
        global_std = np.std(neuron)
        threshold = global_mean + 2 * global_std
        d = [neuron[i * sample_num: i * sample_num + sample_num] for i in range(class_num)]
        d = np.array(d)
        local_mean = np.mean(d, axis=1)
        encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]
        if not encode_class == []:
            SIMI_ind.append(i)
            if len(encode_class) == 1:
                SI_ind.append(i)
            else:
                MI_ind.append(i)
    with open(sig_neuron_dir + '/' + layer + '_SI_ind.pkl', 'wb') as f:
        pickle.dump(SI_ind, f)
    with open(sig_neuron_dir + '/' + layer + '_MI_ind.pkl', 'wb') as f:
        pickle.dump(MI_ind, f)

    print('Doing SVM for SIMI...')
    SIMI_acc = SVM_classification(sig_neuron[:, SIMI_ind], label)
    SIMI_acc_dict.update({layer: SIMI_acc})
    print('ID_Accuracy: %d %%' % (100 * SIMI_acc))

with open(sig_neuron_dir + '/SIMI_acc_dict.pkl', 'wb') as f:
    pickle.dump(SIMI_acc_dict, f)

x = layer_list
y = [SIMI_acc_dict[k] for k in layer_list]
plt.figure()
plt.plot(x, y, 'b', label='SIMI')
plt.ylim((0, 1))
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Classification Accuracy')
plt.title('SIMI_neuron Decoding Accuracy')
plt.savefig(sig_neuron_dir + '/SIMI_acc.png')
