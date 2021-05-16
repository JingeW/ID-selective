'''Comparison decoding ability(classification accuracy)
    between ID-selective and Non_ID-selective neuron

    Comparison through all layers
    Save fullMatrix into pkl file
    Use 10-floder cross validation
'''
import os
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold


def makeFullMatrix(layer, root, pkl_dir, sample_num):
    """ Method to group all feature maps from the same layer
        Parameters:
            layer: str, layer name
            root: str, path of the extraced feature
            pkl_dir: str, path to store the full matrix
            sample_num: int, number of sample per class
        Return:
            full_matrix: a matrix contain all the feature maps in the same player
                         each row is the flatten vector of one feature map(channel*w*h)
                         each col is the neuron response to the img (sample_num*class_num)
    """
    ID_list = sorted([root + '/' + f for f in os.listdir(root)])
    full_matrix = np.empty([0])
    for ID in ID_list:
        img_list = sorted([ID + '/' + f for f in os.listdir(ID)])
        for img in img_list:
            # print('Now processing...', img)
            feat_list = sorted([img + '/' + layer + '/' + f for f in os.listdir(img + '/' + layer)])
            merge_feat = np.empty([0])
            for feat in feat_list:
                matrix = np.loadtxt(feat, delimiter=',').flatten()
                merge_feat = np.concatenate((merge_feat, matrix))
            full_matrix = np.concatenate((full_matrix, merge_feat))
    full_matrix = full_matrix.reshape((sample_num, -1))
    print(full_matrix.shape)
    with open(pkl_dir + '/' + layer + '_fullMatrix.pkl', 'wb') as f:
        pickle.dump(full_matrix, f, protocol=4)
    return full_matrix


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
        d = [neuron[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
        p = stats.f_oneway(d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],  d[8],  d[9],
                           d[10],  d[11],  d[12],  d[13],  d[14],  d[15],  d[16],  d[17],  d[18],  d[19],
                           d[20],  d[21],  d[22],  d[23],  d[24],  d[25],  d[26],  d[27],  d[28],  d[29],
                           d[30],  d[31],  d[32],  d[33],  d[34],  d[35],  d[36],  d[37],  d[38],  d[39],
                           d[40],  d[41],  d[42],  d[43],  d[44],  d[45],  d[46],  d[47],  d[48],  d[49])[1]
        pl.append(p)
    pl = np.array(pl)
    sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
    return sig_neuron_ind


def main():
    # root = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_feat_mat_full'
    root = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_random_feat_mat_full_pkl'
    # pkl_dir = '/home/sdb1/Jinge/ID_selective/features_pkl'
    pkl_dir = '/home/sdb1/Jinge/ID_selective/random_features_pkl'
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_seperate'
    if not os.path.exists(dest):
        os.makedirs(dest)
    layer_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_Vggface_feat_mat_full/01/002475'), reverse=True)
    print('layer_list:', layer_list)
    sample_num = 50 * 10
    num_per_class = 10
    alpha = 0.01
    full_acc_dict = {}
    ID_acc_dict = {}
    nonID_acc_dict = {}

    for layer in layer_list:
        print('****************************')
        print('Building feature matrix of layer', layer + ':')
        full_matrix = makeFullMatrix(layer, root, pkl_dir, sample_num)
        print('Doing ANOVA on feature matrix of layer', layer + ':')
        sig_neuron_ind = oneWay_ANOVA(full_matrix, num_per_class, alpha)

        # Gnerate a boolean mask to return the rest of cols
        mask = np.array([(i in sig_neuron_ind) for i in range(full_matrix.shape[1])])
        ID_matrix = full_matrix[:, mask]
        nonID_matrix = full_matrix[:, ~mask]

        print('Doing classification of layer', layer + ':')
        label = []
        for i in range(50):
            label += [i + 1] * 10
        label = np.array(label)
        label = np.expand_dims(label, axis=1)
        full_matrix = np.concatenate((full_matrix, label), axis=1)
        np.random.shuffle(full_matrix)
        ID_matrix = np.concatenate((ID_matrix, label), axis=1)
        np.random.shuffle(ID_matrix)
        nonID_matrix = np.concatenate((nonID_matrix, label), axis=1)
        np.random.shuffle(nonID_matrix)
        samples = int(sample_num / 10)
        kf = KFold(n_splits=10)

        # KFold for fullMatrix
        print('KFold for fullMatrix...')
        split_index = kf.split(full_matrix)
        clf = svm.SVC()
        full_acc_list = []
        for full_train, full_test in split_index:
            full_pred = clf.fit(full_matrix[full_train, : -1], full_matrix[full_train, -1]
                                ).predict(full_matrix[full_test, : -1])
            full_correct = 0
            for i in range(samples):
                if full_pred[i] == full_matrix[full_test, -1][i]:
                    full_correct += 1
            full_acc = full_correct / samples
            print('full_Accuracy: %d %%' % (100 * full_acc))
            full_acc_list.append(full_acc)
        Avg_full_acc = sum(full_acc_list) / len(full_acc_list)
        full_acc_dict.update({layer: Avg_full_acc})
        print('Avg_full_acc: %d %%' % (Avg_full_acc * 100))

        # KFold for ID_Matrix
        print('KFold for ID_Matrix...')
        split_index1 = kf.split(ID_matrix)
        clf1 = svm.SVC()
        ID_acc_list = []
        for ID_train, ID_test in split_index1:
            ID_pred = clf1.fit(ID_matrix[ID_train, : -1], ID_matrix[ID_train, -1]
                               ).predict(ID_matrix[ID_test, : -1])
            ID_correct = 0
            for i in range(samples):
                if ID_pred[i] == ID_matrix[ID_test, -1][i]:
                    ID_correct += 1
            ID_acc = ID_correct / samples
            print('ID_Accuracy: %d %%' % (100 * ID_acc))
            ID_acc_list.append(ID_acc)
        Avg_ID_acc = sum(ID_acc_list) / len(ID_acc_list)
        ID_acc_dict.update({layer: Avg_ID_acc})
        print('Avg_ID_acc: %d %%' % (Avg_ID_acc * 100))

        # KFold for nonID_Matrix
        print('KFold for nonID_Matrix...')
        split_index2 = kf.split(nonID_matrix)
        clf2 = svm.SVC()
        nonID_acc_list = []
        for nonID_train, nonID_test in split_index2:
            nonID_pred = clf2.fit(nonID_matrix[nonID_train, : -1], nonID_matrix[nonID_train, -1]
                                  ).predict(nonID_matrix[nonID_test, : -1])
            nonID_correct = 0
            for i in range(samples):
                if nonID_pred[i] == nonID_matrix[nonID_test, -1][i]:
                    nonID_correct += 1
            nonID_acc = nonID_correct / samples
            print('nonID_Accuracy: %d %%' % (100 * nonID_acc))
            nonID_acc_list.append(nonID_acc)
        Avg_nonID_acc = sum(nonID_acc_list) / len(nonID_acc_list)
        nonID_acc_dict.update({layer: Avg_nonID_acc})
        print('Avg_nonID_acc: %d %%' % (Avg_nonID_acc * 100))

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
    plt.plot(x, y2, 'g', label='nonID')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('Decoding Accuracy')
    plt.savefig(dest + '/' + 'acc.png')


if __name__ == "__main__":
    main()
