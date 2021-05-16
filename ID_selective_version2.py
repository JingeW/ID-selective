import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle


def layer_name(string):
    name = string.split('/')[-1].split('.')[0].split('_')
    check_list = ['1', '2', '3']
    if name[4] in check_list:
        layer = name[3] + '_' + name[4]
    else:
        layer = name[3]
    return layer


def make_fullMatrix(path):
    layer = layer_name(path)
    # print('Building feature matrix of layer', name + ':')
    full_matrix = np.loadtxt(path, delimiter=',')
    full_matrix = np.transpose(full_matrix)
    # print('shape of full_matrix:', full_matrix.shape)
    return layer, full_matrix

# def oneWay_ANOVA(layer_name, matrix, sample_num, dest, alpha):
#     row, col = matrix.shape
#     pl = []
#     for i in range(col):
#         neuron = matrix[:, i]
#         d = [neuron[i*sample_num:i*sample_num + sample_num] for i in range(int(row/sample_num))]
#         p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
#                             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
#                             d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
#                             d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
#                             d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49] )[1]
#         pl.append(p)
#     pl = np.array(pl)
#     np.savetxt(dest + '/' + layer_name + '_plist.csv', pl, delimiter=',')
#     sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
#     print(layer_name, 'has', len(sig_neuron_ind), 'significant neurons')
#     return sig_neuron_ind


def oneWay_ANOVA(matrix, num_per_class, alpha):
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
    full_matrix_root = '/home/sdb1/Jinge/ID_selective/features_csv'
    stats_dest = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface'
    if not os.path.exists(stats_dest):
        os.makedirs(stats_dest)
    num_per_class = 10
    alpha = 0.01
    full_matrix_path_list = [full_matrix_root + '/' + f for f in os.listdir(full_matrix_root)]
    cnt = []
    percent_list = []
    count = {}
    percent_dict = {}

    for full_matrix_path in sorted(full_matrix_path_list):
        layer_name, full_matrix = make_fullMatrix(full_matrix_path)
        print('Shape of', layer_name, ':', full_matrix.shape)
        print('Doing ANOVA on feature matrix of layer', layer_name + ':')
        sig_neuron_ind,  num,  percent = oneWay_ANOVA(full_matrix,  num_per_class,  alpha)
        print(layer_name, 'has', num, 'significant neurons')
        print('Percentage: {:.2f}%'.format(percent))
        sig_neuron = full_matrix[:,  sig_neuron_ind]
        print('Writing pkl file for', layer_name)
        with open(stats_dest + '/' + layer_name + '_sig_neuron.pkl',  'wb') as f:
            pickle.dump(sig_neuron, f, protocol=4)
        cnt.append(num)
        percent_list.append(percent)
        count.update({layer_name: num})
        percent_dict.update({layer_name: percent})
    cnt = np.array(cnt)
    percent_list = np.array(percent_list)
    np.savetxt(stats_dest + '/' + 'count.csv', cnt, delimiter=', ')
    np.savetxt(stats_dest + '/' + 'percent.csv', percent_list, delimiter=', ')

    # plot
    layer_squence = [
        'conv1_1',  'conv1_2',  'pool1',
        'conv2_1',  'conv2_2',  'pool2',
        'conv3_1',  'conv3_2',  'conv3_3',  'pool3',
        'conv4_1',  'conv4_2',  'conv4_3',  'pool4',
        'conv5_1',  'conv5_2',  'conv5_3',  'pool5',
        'fc6',  'fc7',  'fc8'
    ]
    x_1 = layer_squence
    y_1 = [count[k] for k in layer_squence]
    plt.figure(0)
    plt.bar(x_1, y_1, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts')
    plt.title('Significant neuron counts for different layers')
    plt.savefig(stats_dest + '/' + 'count.png')
    # plt.show()

    x_2 = layer_squence
    y_2 = [percent_dict[k] for k in layer_squence]
    plt.figure(1)
    plt.bar(x_2, y_2, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron percentage')
    plt.title('Significant neuron percentage for different layers')
    plt.savefig(stats_dest + '/' + 'percent.png')
    # plt.show()


if __name__ == "__main__":
    main()
