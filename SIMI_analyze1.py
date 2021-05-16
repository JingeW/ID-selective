''' Generate SIMI mask for fullMatrix
'''
import pickle

sig_neuron_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
SIMI_dir = '/home/sdb1/Jinge/ID_selective/Stats/VGG16_Vggface_full'
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]

for layer in layer_list:
    print('Now processing layer:', layer)
    sig_neuron_ind_path = sig_neuron_dir + '/' + layer + '_sig_neuron_ind.pkl'
    SI_path = SIMI_dir + '/' + layer + '_SI_ind.pkl'
    MI_path = SIMI_dir + '/' + layer + '_MI_ind.pkl'

    with open(sig_neuron_ind_path, 'rb') as f:
        sig_neuron_ind = pickle.load(f)
    with open(SI_path, 'rb') as f:
        SI = pickle.load(f)
    with open(MI_path, 'rb') as f:
        MI = pickle.load(f)

    SI_ind = set([sig_neuron_ind[i] for i in SI])
    MI_ind = set([sig_neuron_ind[i] for i in MI])
    SIMI_ind = list(SI_ind | MI_ind)

    with open(sig_neuron_dir + '/' + layer + '_SIMI_ind.pkl', 'wb') as f:
        pickle.dump(SIMI_ind, f)
