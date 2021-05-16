'''One-stop ID selective neuron analysis
    Input: feature maps of the set of images
    Output: Accuracy plot of each layer for the set of image with ID/nonID neuron mask embedded
    TODO:
    1. Generate full matrix: Assemble all channel's feature maps of all images for each layer
    2. Generate mask: Do Anova test cross ID for each neuron in each layer
    3. Generate Acc plot: Do SVM of full/ID/nonID neurons for each layer
'''

import os
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


def makeFullMatrix(layer, featureMap_path, sample_num, class_num):
    """ Method to group all feature maps from the same layer
        Parameters:
            layer: str, layer name
            featureMap_path: str, path of the extraced feature
            sample_num: int, number of sample per class
            class_num: int, number of class
        Return:
            fullMatrix: a matrix contain all the feature maps in the same player
                         each row is the flatten vector of one feature map(channel*w*h)
                         each col is the neuron response to the img (sample_num*class_num)
    """
    ID_list = sorted([featureMap_path + '/' + f for f in os.listdir(featureMap_path)])
    fullMatrix = np.empty([0])
    for ID in ID_list:
        img_list = sorted([ID + '/' + f for f in os.listdir(ID)])
        for img in img_list:
            feat_list = sorted([img + '/' + layer + '/' + f for f in os.listdir(img + '/' + layer)])
            merge_feat = np.empty([0])
            for feat in feat_list:
                f = open(feat, 'rb')
                matrix = pickle.load(f)
                f.close
                matrix = matrix.flatten()
                merge_feat = np.concatenate((merge_feat, matrix))
            fullMatrix = np.concatenate((fullMatrix, merge_feat))
    row_num = sample_num * class_num
    fullMatrix = fullMatrix.reshape((row_num, -1))
    return fullMatrix


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


def makeLabels(sample_num, class_num):
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label


def makeFolder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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


if __name__ == "__main__":
    featureMap_path = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_cartoonFace_random_featureMaps'
    name = featureMap_path.split('/')[-1][:-12]
    mask_path = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
    fullMatrix_path = '/home/sdb1/Jinge/ID_selective/fullFM/' + name
    makeFolder(fullMatrix_path)
    result_path = '/home/sdb1/Jinge/ID_selective/Result/' + name
    makeFolder(result_path)
    sample_num = 10
    class_num = 50
    label = makeLabels(sample_num, class_num)
    alpha = 0.01
    full_acc_dict = {}
    ID_acc_dict = {}
    nonID_acc_dict = {}
    layer_list = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
        'FC6', 'FC7', 'FC8'
    ]

    for layer in layer_list:
        print('****************************')
        print('Now working on layer:', layer)

        print('1. Building feature matrix...')
        # featureMap to fullMatrix
        fullMatrix = makeFullMatrix(layer, featureMap_path, sample_num, class_num)
        # write fullMatrix to pkl
        with open(fullMatrix_path + '/' + layer + '_fullMatrix.pkl', 'wb') as f:
            pickle.dump(fullMatrix, f, protocol=4)
        print('Shape of full matrix:', fullMatrix.shape)

        print('2. Doing ANOVA on feature matrix of layer...')
        # generate mask of current input by ANOVA
        sig_neuron_ind, non_sig_neuron_ind = oneWay_ANOVA(fullMatrix, sample_num, class_num, alpha)
        # write mask to pkl
        with open(result_path + '/' + layer + '_sig_neuron_ind.pkl', 'wb') as f:
            pickle.dump(sig_neuron_ind, f)
        with open(result_path + '/' + layer + '_non_sig_neuron_ind.pkl', 'wb') as f:
            pickle.dump(non_sig_neuron_ind, f)

        print('3. Doing classification of layer...')
        # SVM for full/ID/nonID
        print('Doing SVM for full...')
        full_acc = SVM_classification(fullMatrix, label)
        full_acc_dict.update({layer: full_acc})
        print('full_Accuracy: %d %%' % (100 * full_acc))

        print('Loading mask...')
        # Load celebA original mask
        with open(mask_path + '/' + layer + '_sig_neuron_ind.pkl', 'rb') as f:
            maskID = pickle.load(f)
        with open(mask_path + '/' + layer + '_non_sig_neuron_ind.pkl', 'rb') as f:
            maskNonID = pickle.load(f)
        print('Length of ID/nonID mask:', len(maskID), len(maskNonID))

        print('Doing SVM for ID...')
        ID_acc = SVM_classification(fullMatrix[:, maskID], label)
        ID_acc_dict.update({layer: ID_acc})
        print('ID_Accuracy: %d %%' % (100 * ID_acc))

        print('Doing SVM for nonID...')
        nonID_acc = SVM_classification(fullMatrix[:, maskNonID], label)
        nonID_acc_dict.update({layer: nonID_acc})
        print('nonID_Accuracy: %d %%' % (100 * nonID_acc))
        del fullMatrix

    with open(result_path + '/full_acc_dict.pkl', 'wb') as f:
        pickle.dump(full_acc_dict, f)
    with open(result_path + '/ID_acc_dict.pkl', 'wb') as f:
        pickle.dump(ID_acc_dict, f)
    with open(result_path + '/nonID_acc_dict.pkl', 'wb') as f:
        pickle.dump(nonID_acc_dict, f)
    print('Accuracy dict saved!')

    x = layer_list
    y = [full_acc_dict[k] for k in layer_list]
    y1 = [ID_acc_dict[k] for k in layer_list]
    y2 = [nonID_acc_dict[k] for k in layer_list]
    plt.figure()
    plt.plot(x, y, 'b', label='full')
    plt.plot(x, y1, 'r', label='ID')
    plt.plot(x, y2, 'g', label='nonID')
    plt.ylim((0, 1))
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('Decoding Accuracy')
    plt.savefig(result_path + '/' + name + '_acc.png')
