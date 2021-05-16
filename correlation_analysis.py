import os
import pickle
import numpy as np
from scipy.io import savemat
from scipy.spatial.distance import pdist, squareform
import seaborn as sn
import matplotlib.pyplot as plt


def avg_acrossID(matrix, sample_num, class_num):
    col = matrix.shape[1]
    avg_full = np.empty((class_num, col))
    for i in range(class_num):
        subMat = matrix[i * sample_num: i * sample_num + sample_num, :]
        avg_sub = subMat.mean(axis=0)  # to take the mean of each col
        avg_full[i, :] = avg_sub
    return avg_full


# *********Adjustable***********
dataSet_name = 'CelebA_original'
sample_num = 10
class_num = 50
# ******************************

print('=========', 'Correlation_analysis for', dataSet_name, '=========')
root = '/home/sdb1/Jinge/ID_selective/fullFM32/VGG16_Vggface_' + dataSet_name
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
dest = '/home/sdb1/Jinge/ID_selective/Result/CorAndDist/' + dataSet_name
if not os.path.exists(dest):
    os.makedirs(dest)

layer_list = [
    # 'Conv1_1', 'Conv1_2', 'Pool1',
    # 'Conv2_1', 'Conv2_2', 'Pool2',
    # 'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    # 'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    # 'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]

# cor_full_dict = {}
# cor_full_ID_dict = {}
# cor_full_nonID_dict = {}
# cor_avg_dict = {}
# cor_avg_ID_dict = {}
# cor_avg_nonID_dict = {}
# dist_full_dict = {}
# dist_avg_dict = {}

for layer in layer_list:
    print('****************************')
    print('Now working on layer:', layer)

    # Full: 500*col
    print('Loading feature matrix...')
    with open(root + '/' + layer + '_fullMatrix.pkl', 'rb') as f:
        fullMatrix = pickle.load(f)

    # Avg: 50*col
    avgMatrix = avg_acrossID(fullMatrix, sample_num, class_num)

    # Mask: ID, nonID
    print('Loading mask...')
    with open(mask_dir + '/' + layer + '_sig_neuron_ind.pkl', 'rb') as f:
        maskID = pickle.load(f)
    with open(mask_dir + '/' + layer + '_non_sig_neuron_ind.pkl', 'rb') as f:
        maskNonID = pickle.load(f)
    print('Length of ID/nonID mask:', len(maskID), len(maskNonID))

    # # 1. full correlation matrix
    # cor_full = np.corrcoef(fullMatrix)
    # cor_full_dict.update({layer: cor_full})

    # 2. full ID/nonID correlation matrix
    cor_full_ID = np.corrcoef(fullMatrix[:, maskID])
    # cor_full_ID_dict.update({layer: cor_full_ID})
    cor_full_nonID = np.corrcoef(fullMatrix[:, maskNonID])
    # cor_full_nonID_dict.update({layer: cor_full_nonID})

    # # 3. avg correlation matrix
    cor_avg = np.corrcoef(avgMatrix)
    # cor_avg_dict.update({layer: cor_avg})

    # 4. avg ID/nonID correlation matri
    cor_avg_ID = np.corrcoef(avgMatrix[:, maskID])
    # cor_avg_ID_dict.update({layer: cor_avg_ID})

    cor_avg_nonID = np.corrcoef(avgMatrix[:, maskNonID])
    # cor_avg_nonID_dict.update({layer: cor_avg_nonID})

    # # 4. full dist matrix
    # dist_full = pdist(fullMatrix, 'euclidean')
    # dist_full_dict.update({layer: dist_full})

    # 5. avg dist matrix
    dist_avg = pdist(avgMatrix, 'euclidean')
    # dist_avg_dict.update({layer: dist_avg})

    m = squareform(dist_avg)
    sn.heatmap(m)
    plt.show()


# savemat(dest + '/' + 'CorMatrix_full.mat', cor_full_dict)
# savemat(dest + '/' + 'CortMatrix_full_ID.mat', cor_full_ID_dict)
# savemat(dest + '/' + 'CorMatrix_full_nonID.mat', cor_full_nonID_dict)
# savemat(dest + '/' + 'CorMatrix_avg.mat', cor_avg_dict)
# savemat(dest + '/' + 'CortMatrix_avg_ID.mat', cor_avg_ID_dict)
# savemat(dest + '/' + 'CorMatrix_avg_nonID.mat', cor_avg_nonID_dict)
# savemat(dest + '/' + 'DistMatrix_full.mat', dist_full_dict)
# savemat(dest + '/' + 'DistMatrix_avg.mat', dist_avg_dict)

# print(layer, 'mat saved!')
