'''
plot for VGG16_Vggface_full
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def make_layer_name(string):
    name = string.split('/')[-1].split('.')[0].split('_')
    check_list = ['1', '2', '3']
    if name[1] in check_list:
        layer_name = name[0] + '_' + name[1]
    else:
        layer_name = name[0]
    return layer_name


def main():
    root = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
    save_path = root + '/Sig_layout'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sigNeuron_list = sorted([os.path.join(root, f) for f in os.listdir(root) if 'neuron' in f.split('_')[-1]])
    cnt = []
    percent_list = []
    count = {}
    percent_dict = {}
    total_neuron = {
        'Conv1_1': 224 * 224 * 64, 'Conv1_2': 224 * 224 * 64, 'Pool1': 112 * 112 * 128,
        'Conv2_1': 112 * 112 * 128, 'Conv2_2': 112 * 112 * 128, 'Pool2': 56 * 56 * 256,
        'Conv3_1': 56 * 56 * 256, 'Conv3_2': 56 * 56 * 256, 'Conv3_3': 56 * 56 * 256, 'Pool3': 28 * 28 * 512,
        'Conv4_1': 28 * 28 * 512, 'Conv4_2': 28 * 28 * 512, 'Conv4_3': 28 * 28 * 512, 'Pool4': 14 * 14 * 512,
        'Conv5_1': 14 * 14 * 512, 'Conv5_2': 14 * 14 * 512, 'Conv5_3': 14 * 14 * 512, 'Pool5': 7 * 7 * 512,
        'FC6': 4096, 'FC7': 4096, 'FC8':  2622,
    }

    for sigNeuron in sigNeuron_list:
        layer_name = make_layer_name(sigNeuron)
        print('Now processing...', layer_name)
        print('Dim :', total_neuron[layer_name])
        with open(sigNeuron, 'rb') as f:
            matrix = pickle.load(f)
            col = len(matrix[1])
            percent = col / total_neuron[layer_name] * 100
            cnt.append(col)
            percent_list.append(percent)
            count.update({layer_name: col})
            percent_dict.update({layer_name: percent})
    cnt = np.array(cnt)
    percent_list = np.array(percent_list)
    # np.savetxt(root + '/' + 'count.csv', cnt, delimiter=',')
    # np.savetxt(root + '/' + 'percent.csv', percent_list, delimiter=',')

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
    plt.figure(1)
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts')
    plt.title('Significant neuron counts for different layers')
    plt.savefig(save_path + '/count.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/count.eps', bbox_inches='tight', dpi=100, format='eps')
    plt.savefig(save_path + '/count.svg', bbox_inches='tight', dpi=100, format='svg')

    # plt.show()

    x = layer_squence
    y = [percent_dict[k] for k in layer_squence]
    plt.figure(2)
    plt.bar(x, y, width=0.5)
    plt.xticks(rotation=45)
    plt.ylabel('sig neuron counts percentage')
    plt.title('Significant neuron counts percentage for different layers')
    plt.savefig(save_path + '/percent.png', bbox_inches='tight', dpi=100)
    plt.savefig(save_path + '/percent.eps', bbox_inches='tight', dpi=100, format='eps')
    plt.savefig(save_path + '/percent.svg', bbox_inches='tight', dpi=100, format='svg')
    # plt.show()


if __name__ == "__main__":
    main()
