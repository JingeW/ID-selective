'''Get the sig_neuron_ind from Observation set
    and plot full acc result
    Generate mask for diff input
'''

import os
import pickle
# import numpy as np
import scipy.stats as stats
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


def oneWay_ANOVA(fullMatrix, sample_num, class_num, alpha):
    ''' ANOVA test for each neuron among all classes
            H0: The testing neuron has the same response to different images
            H1: The testing neuron has at least one different response to different images
        Parameters:
            fullMatrix: array, is the full matrix which consists of all featrue maps in the same layer
            sample_num: int, number of sample per class
            class_num: int, number of class
            alpha: float, significant level
        Return:
            sig_neuron_ind: a list store the index of neuron which ANOVA tested significantly
            non_sig_neuron_ind: a list store the index of neuron which ANOVA tested insignificantly
    '''
    col = fullMatrix.shape[1]
    print('ANOVA iter range:', col)
    sig_neuron_ind = []
    non_sig_neuron_ind = []
    for i in range(col):
        neuron = fullMatrix[:, i]
        d = [neuron[i * sample_num: i * sample_num + sample_num] for i in range(class_num)]
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        if p < alpha:
            sig_neuron_ind.append(i)
        else:
            non_sig_neuron_ind.append(i)
    return sig_neuron_ind, non_sig_neuron_ind


def main():
    root = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_CelebA_original_layer_shuffle'
    dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original_layer_shuffle'
    if not os.path.exists(dest):
        os.makedirs(dest)
    # layer_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_Vggface_feat_mat_full/01/002475'), reverse=True)
    # layer_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_Vggface_feat_mat_full/01/002475'))
    layer_list = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
        'FC6', 'FC7', 'FC8'
    ]
    print('layer_list:', layer_list)
    sample_num = 10
    class_num = 50
    alpha = 0.01
    # full_acc_dict = {}
    # ID_acc_dict = {}
    # nonID_acc_dict = {}

    for layer in layer_list:
        # layer = layer_list[0]
        # Load full feat mat and do ANOVA to filter out sig neuron
        print('****************************')
        print('Loading feature matrix of layer', layer + ':')
        FM_path = os.path.join(root, layer + '_fullMatrix.pkl')
        f = open(FM_path, 'rb')
        full_matrix = pickle.load(f)
        f.close
        print(full_matrix.shape)
        print('Doing ANOVA on feature matrix of layer', layer + ':')
        sig_neuron_ind, non_sig_neuron_ind = oneWay_ANOVA(full_matrix, sample_num, class_num, alpha)

        SNI_path = os.path.join(dest, layer + '_sig_neuron_ind.pkl')
        with open(SNI_path, 'wb') as f:
            pickle.dump(sig_neuron_ind, f)
        print('SNI pkl saved!')

        nonSNI_path = os.path.join(dest, layer + '_non_sig_neuron_ind.pkl')
        with open(nonSNI_path, 'wb') as f:
            pickle.dump(non_sig_neuron_ind, f)
        print('nonSNI pkl saved!')

    #     print('Doing classification of layer', layer + ':')
    #     # Create label for feat mat
    #     label = []
    #     for i in range(50):
    #         label += [i + 1] * 10
    #     label = np.array(label)
    #     label = np.expand_dims(label, axis=1)
    #     print('Label generated!')

    #     # SVM for full
    #     full_train, full_test, label_train, label_test = train_test_split(full_matrix, label, test_size=0.33, random_state=42)
    #     print('full data load complete, doing SVM...')
    #     clf = svm.SVC()
    #     clf.fit(full_train, label_train.ravel())
    #     full_predicted = clf.predict(full_test)
    #     full_correct = 0
    #     samples = len(label_test)
    #     for i in range(samples):
    #         if full_predicted[i] == label_test[i]:
    #             full_correct += 1
    #     full_acc = full_correct / samples
    #     print('full_Accuracy: %d %%' % (100 * full_acc))
    #     full_acc_dict.update({layer: full_acc})
    #     del full_train, full_test, label_train, label_test
    #     print('full data SVM complete, memory released.')

    #     # Gnerate a boolean mask to return the rest of cols
    #     mask = np.array([(i in sig_neuron_ind) for i in range(full_matrix.shape[1])])
    #     # ID_matrix = full_matrix[:, mask] ###
    #     # nonID_matrix = full_matrix[:, ~mask] ###
    #     # print('ID & nonID matrix generated!')
    #     print('mask generated!')
    #     # del full_matrix ###

    #     # SVM for ID
    #     # ID_train, ID_test, label_train, label_test = train_test_split(ID_matrix, label, test_size=0.33, random_state=42) ###
    #     ID_train, ID_test, label_train, label_test = train_test_split(full_matrix[:, mask], label, test_size=0.33, random_state=42)
    #     print('ID data load complete, doing SVM...')
    #     clf1 = svm.SVC()
    #     clf1.fit(ID_train, label_train.ravel())
    #     ID_predicted = clf1.predict(ID_test)
    #     ID_correct = 0
    #     for i in range(samples):
    #         if ID_predicted[i] == label_test[i]:
    #             ID_correct += 1
    #     ID_acc = ID_correct / samples
    #     print('ID_Accuracy: %d %%' % (100 * ID_acc))
    #     ID_acc_dict.update({layer: ID_acc})
    #     del ID_train, ID_test, label_train, label_test
    #     # del ID_matrix ###
    #     print('ID data SVM complete, memory released.')

    #     # SVM for non-ID
    #     # nonID_train, nonID_test, label_train, label_test = train_test_split(nonID_matrix, label, test_size=0.33, random_state=42) ###
    #     nonID_train, nonID_test, label_train, label_test = train_test_split(full_matrix[:, ~mask], label, test_size=0.33, random_state=42)
    #     print('nonID data load complete, doing SVM...')
    #     clf2 = svm.SVC()
    #     clf2.fit(nonID_train, label_train.ravel())
    #     nonID_predicted = clf2.predict(nonID_test)
    #     nonID_correct = 0
    #     for i in range(samples):
    #         if nonID_predicted[i] == label_test[i]:
    #             nonID_correct += 1
    #     nonID_acc = nonID_correct / samples
    #     print('nonID_Accuracy: %d %%' % (100 * nonID_acc))
    #     nonID_acc_dict.update({layer: nonID_acc})
    #     del nonID_train, nonID_test, label_train, label_test
    #     # del nonID_matrix ###
    #     del full_matrix
    #     print('nonID data SVM complete, memory released.')

    # # Plot
    # layer_squence = [
    #     'Conv1_1', 'Conv1_2', 'Pool1',
    #     'Conv2_1', 'Conv2_2', 'Pool2',
    #     'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    #     'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    #     'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    #     'FC6', 'FC7', 'FC8'
    # ]

    # x = layer_squence
    # y = [full_acc_dict[k] for k in layer_squence]
    # y1 = [ID_acc_dict[k] for k in layer_squence]
    # y2 = [nonID_acc_dict[k] for k in layer_squence]
    # plt.figure()
    # plt.plot(x, y, 'b', label='full')
    # plt.plot(x, y1, 'r', label='ID')
    # plt.plot(x, y2, 'g', label='nonID')
    # plt.legend()
    # plt.xticks(rotation=45)
    # plt.ylabel('Classification Accuracy')
    # plt.title('Decoding Accuracy')
    # plt.savefig(dest + '/' + 'acc.png')


if __name__ == "__main__":
    main()
