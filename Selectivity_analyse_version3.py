'''Comparison decoding ability(classification accuracy)
    between ID-selective and Non_ID-selective neuron

    TODO:
        1. Seperate ID and Non-ID feature for each layer
            use binary mask
        2. Pick a certain layer to do the classification
            tried Conv5_3, FC8
            and the rest of layers
        3. Pick a method to do the classification
            SVM
        4. Compare and plot the accuracy between ID and Non-ID selective neuron
'''

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
# import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold


def layer_name(string):
    name = string.split('.')[0].split('_')
    check_list = ['1', '2', '3']
    if name[2] in check_list:
        layer = name[1] + '_' + name[2]
    else:
        layer = name[1]
    return layer


def oneWay_ANOVA(matrix, num_per_class, alpha):
    ''' ANOVA test for each neuron among all classes
            H0: The testing neuron has the same response to different images
            H1: The testing neuron has at least one different response to different images
        Parameters:
            matrix: array/ np array, is the full matrix which consists of all featrue maps in the same layer
            num_per_class: int, number of sample per class
            alpha: float, significant level
        Return:
            sig_neuron_ind: a list store the index of neuron which ANOVA tested significantly
            num: the number of significant neurons
            percent: num of significant neuron / total number of neuron is that layer
    '''
    row, col = matrix.shape
    pl = []
    for i in range(col):
        neuron = matrix[:, i]
        d = [neuron[i * num_per_class: i * + num_per_class] for i in range(int(row / num_per_class))]
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        pl.append(p)
        pl.append(p)
    pl = np.array(pl)
    sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
    # num = len(sig_neuron_ind)
    # percent = num/col*100
    return sig_neuron_ind


def main():
    fullMatrix_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_full'
    dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_seperate'
    if not os.path.exists(dest):
        os.makedirs(dest)

    fullMatrix_list = sorted(os.listdir(fullMatrix_dir))
    target_list = ['Conv5_3', 'FC8']
    # target_list = ['FC8']
    # target_list = []
    # sample_num = 50 * 10
    num_per_class = 10
    alpha = 0.01
    full_acc_dict = {}
    ID_acc_dict = {}
    nonID_acc_dict = {}

    for matrix in fullMatrix_list:
        layer = layer_name(matrix)
        # if not layer in target_list:
        if layer in target_list:
            fullMatrix_path = os.path.join(fullMatrix_dir, matrix)
            print('****************************')
            print('Loading feature matrix of layer', layer + ':')
            full_matrix = np.loadtxt(fullMatrix_path, delimiter=',')
            print('Doing ANOVA on feature matrix of layer', layer + ':')
            sig_neuron_ind = oneWay_ANOVA(full_matrix, num_per_class, alpha)
            # Gnerate a boolean mask to return the rest of cols
            mask = np.array([(i in sig_neuron_ind) for i in range(full_matrix.shape[1])])
            ID_matrix = full_matrix[:, mask]
            nonID_matrix = full_matrix[:, ~mask]
            # with open(dest+'/'+layer+'_ID_matrix.pkl','wb') as f:
            #     pickle.dump(ID_matrix,f)
            # print('ID_matrix saved')
            # with open(dest+'/'+layer+'_nonID_matrix.pkl','wb') as f:
            #     pickle.dump(nonID_matrix,f)
            # print('nonID_matrix saved')
            # print(layer+'->','Shape of ID_matrix:', ID_matrix.shape)
            # print(layer+'->','Shape of nonID_matrix:', nonID_matrix.shape)

            label = []
            for i in range(50):
                label += [i + 1] * 10
            label = np.array(label)
            label = np.expand_dims(label, axis=1)

            # full_matrix = np.concatenate((full_matrix, label),axis=1)
            # np.random.shuffle(full_matrix)
            # ID_matrix = np.concatenate((ID_matrix, label),axis=1)
            # np.random.shuffle(ID_matrix)
            # nonID_matrix = np.concatenate((nonID_matrix, label),axis=1)
            # np.random.shuffle(nonID_matrix)
            # samples = int(sample_num / 10)
            # kf = KFold(n_splits=10)

            # # KFold for fullMatrix
            # print('KFold for fullMatrix...')
            # split_index = kf.split(full_matrix)
            # clf = svm.SVC()
            # full_acc_list = []

            # for full_train, full_test in split_index:
            #     full_pred = clf.fit(full_matrix[full_train,:-1],full_matrix[full_train,-1]
            #     ).predict(full_matrix[full_test,:-1])
            #     full_correct = 0
            #     for i in range(samples):
            #         if full_pred[i] == full_matrix[full_test,-1][i]:
            #             full_correct += 1
            #     full_acc = full_correct / samples
            #     print('full_Accuracy: %d %%' % (100*full_acc))
            #     full_acc_list.append(full_acc)
            # Avg_full_acc = sum(full_acc_list)/len(full_acc_list)
            # full_acc_dict.update({layer:Avg_full_acc})
            # print('Avg_full_acc: %d %%' % (Avg_full_acc*100))

            # # KFold for ID_Matrix
            # print('KFold for ID_Matrix...')
            # split_index1 = kf.split(ID_matrix)
            # clf1 = svm.SVC()
            # ID_acc_list = []
            # for ID_train, ID_test in split_index1:
            #     ID_pred = clf1.fit(ID_matrix[ID_train,:-1],ID_matrix[ID_train,-1]
            #     ).predict(ID_matrix[ID_test,:-1])
            #     ID_correct = 0
            #     for i in range(samples):
            #         if ID_pred[i] == ID_matrix[ID_test,-1][i]:
            #             ID_correct += 1
            #     ID_acc = ID_correct / samples
            #     print('ID_Accuracy: %d %%' % (100*ID_acc))
            #     ID_acc_list.append(ID_acc)
            # Avg_ID_acc = sum(ID_acc_list)/len(ID_acc_list)
            # ID_acc_dict.update({layer:Avg_ID_acc})
            # print('Avg_ID_acc: %d %%' % (Avg_ID_acc*100))

            # # KFold for nonID_Matrix
            # print('KFold for nonID_Matrix...')
            # split_index2 = kf.split(nonID_matrix)
            # clf2 = svm.SVC()
            # nonID_acc_list = []
            # for nonID_train, nonID_test in split_index2:
            #     nonID_pred = clf2.fit(nonID_matrix[nonID_train,:-1],nonID_matrix[nonID_train,-1]
            #     ).predict(nonID_matrix[nonID_test,:-1])
            #     nonID_correct = 0
            #     for i in range(samples):
            #         if nonID_pred[i] == nonID_matrix[nonID_test,-1][i]:
            #             nonID_correct += 1
            #     nonID_acc = nonID_correct / samples
            #     print('nonID_Accuracy: %d %%' % (100*nonID_acc))
            #     nonID_acc_list.append(nonID_acc)
            # Avg_nonID_acc = sum(nonID_acc_list)/len(nonID_acc_list)
            # nonID_acc_dict.update({layer:Avg_nonID_acc})
            # print('Avg_nonID_acc: %d %%' % (Avg_nonID_acc*100))

            full_train, full_test, label_train, label_test = train_test_split(full_matrix, label, test_size=0.33, random_state=42)
            ID_train, ID_test, label_train, label_test = train_test_split(ID_matrix, label, test_size=0.33, random_state=42)
            nonID_train, nonID_test, label_train, label_test = train_test_split(nonID_matrix, label, test_size=0.33, random_state=42)
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

    # Plot
    # layer_squence = [
    #     'Conv1_1', 'Conv1_2', 'Pool1',
    #     'Conv2_1', 'Conv2_2', 'Pool2',
    #     'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    #     'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    #     'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    #     'FC6', 'FC7', 'FC8'
    #     ]

    layer_squence = ['Conv5_3', 'FC8']
    x = layer_squence
    y = [full_acc_dict[k] for k in layer_squence]
    y1 = [ID_acc_dict[k] for k in layer_squence]
    y2 = [nonID_acc_dict[k] for k in layer_squence]
    plt.figure()
    plt.plot(x, y, 'b', label='full')
    plt.plot(x, y1, 'r', label='ID')
    plt.plot(x, y2, 'b', label='full')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('Decoding Accuracy')
    plt.savefig(dest + '/' + 'acc.png')


if __name__ == "__main__":
    main()
