'''
Write to functions
Introduce pickle to save large data
'''
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle


def makeFullMatrix(layer, root, dest, sample_num):
    """ Method to group all feature maps from the same layer
        Parameters:
            layer: str, layer name
            root: str, path of the extraced feature
            dest: str, path to store the full matrix
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
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        pl.append(p)
    pl = np.array(pl)
    sig_neuron_ind = [ind for ind,  p in enumerate(pl) if p < alpha]
    num = len(sig_neuron_ind)
    percent = num / col * 100
    return sig_neuron_ind,  num,  percent


def main():
    root = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_random_feat_mat_full_pkl'
    dest = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_random_full'
    if not os.path.exists(dest):
        os.makedirs(dest)
    layer_list = sorted(os.listdir('/home/sdb1/Jinge/ID_selective/VGG16_Vggface_feat_mat_full/01/002475'))
    print('layer_list:', layer_list)
    sample_num = 50 * 10
    stats_dest = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
    if not os.path.exists(stats_dest):
        os.makedirs(stats_dest)
    num_per_class = 10
    alpha = 0.01
    cnt = []
    percent_list = []
    count = {}
    percent_dict = {}

    for layer in layer_list:
        # check_list = os.listdir('/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full')
        # if layer+'_'+'sig_neuron.pkl' in check_list:
        #     continue
        print('****************************')
        print('Building feature matrix of layer', layer + ':')
        full_matrix = makeFullMatrix(layer, root, dest, sample_num)
        print('Doing ANOVA on feature matrix of layer', layer + ':')
        sig_neuron_ind, num, percent = oneWay_ANOVA(full_matrix, num_per_class, alpha)
        print(layer, 'has', num, 'significant neurons')
        print('Percentage: {:.2f}%'.format(percent))
        sig_neuron = full_matrix[:, sig_neuron_ind]
        print('Writing pkl file for', layer)
        with open(stats_dest + '/' + layer + '_sig_neuron.pkl', 'wb') as f:
            pickle.dump(sig_neuron, f, protocol=4)
        # np.savetxt(stats_dest+'/'+layer_name+'_sig_neuron.csv',sig_neuron,delimiter=',')
        cnt.append(num)
        percent_list.append(percent)
        count.update({layer: num})
        percent_dict.update({layer: percent})
    cnt = np.array(cnt)
    percent_list = np.array(percent_list)
    np.savetxt(stats_dest + '/' + 'count.csv', cnt, delimiter=',')
    np.savetxt(stats_dest + '/' + 'percent.csv', percent_list, delimiter=',')

    # plot
    layer_squence = [
        'Conv1_1', 'Conv1_2', 'Pool1',
        'Conv2_1', 'Conv2_2', 'Pool2',
        'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
        'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
        'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
        'FC6', 'FC7', 'FC8'
    ]
    x = layer_squence
    y = [count[k] for k in layer_squence]
    plt.figure(0)
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts')
    plt.title('Significant neuron counts for different layers')
    plt.savefig(stats_dest + '/' + 'count.png')
    # plt.show()

    x = layer_squence
    y = [percent_dict[k] for k in layer_squence]
    plt.figure(1)
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts percentage')
    plt.title('Significant neuron counts percentage for different layers')
    plt.savefig(stats_dest + '/' + 'percent.png')
    # plt.show()


if __name__ == "__main__":
    main()
