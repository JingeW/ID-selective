''' Compute Euclidean Distance between IDs for different layers
    respect to 3 grps, Full/ ID/ nonID
    TODO:
    1. Convert fullMatrix to avgMatrix
    2. Compute Euclidean Distance Matrix for each avgMarix (50 * 50)
    3. Vectorize the Distance Matrix (1225 * 1)
    4. Combine all layers (1225 * 21)
'''

import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.io import savemat

root = '/home/sdb1/Jinge/ID_selective/Result/Distance'
FM_dir = '/home/sdb1/Jinge/ID_selective/fullFM/fullFM_CelebA_original_pkl'
mask_dir = '/home/sdb1/Jinge/ID_selective/Result/VGG16_Vggface_CelebA_original'
layer_list = [
    'Conv1_1', 'Conv1_2', 'Pool1',
    'Conv2_1', 'Conv2_2', 'Pool2',
    'Conv3_1', 'Conv3_2', 'Conv3_3', 'Pool3',
    'Conv4_1', 'Conv4_2', 'Conv4_3', 'Pool4',
    'Conv5_1', 'Conv5_2', 'Conv5_3', 'Pool5',
    'FC6', 'FC7', 'FC8'
]
num_per_class = 10
num_of_ID = 50
DistMat_Full = {}
DistMat_ID = {}
DistMat_nonID = {}

#  === 1. Convert fullMatrix to avgMatrix ===
for layer in layer_list:
    print('********************')
    print('Processing:', layer)

    print('Loading fullMatrix...')
    FM_path = FM_dir + '/' + layer + '_fullMatrix.pkl'
    with open(FM_path, 'rb') as f:
        fullMatrix = pickle.load(f)
    num_of_neuron_Full = fullMatrix.shape[-1]

    print('Loading mask...')
    maskID_path = mask_dir + '/' + layer + '_sig_neuron_ind.pkl'
    with open(maskID_path, 'rb') as f:
        mask_ID = pickle.load(f)
    maskNonID_path = mask_dir + '/' + layer + '_non_sig_neuron_ind.pkl'
    with open(maskNonID_path, 'rb') as f:
        mask_nonID = pickle.load(f)

    print('Coverting to avgMatrix...')
    avgMatrix_Full = []
    for ind in range(num_of_neuron_Full):
        neuron_vector = fullMatrix[:, ind]
        ID_vector_list = [neuron_vector[k * num_per_class: k * num_per_class + num_per_class] for k in range(num_of_ID)]
        ID_vector_list = np.array(ID_vector_list)
        mean_list = np.mean(ID_vector_list, axis=1)
        avgMatrix_Full.append(mean_list)
    avgMatrix_Full = np.array(avgMatrix_Full).transpose()
    print(avgMatrix_Full.shape)

    avgMatrix_ID = []
    for ind_ID in mask_ID:
        neuron_vector = fullMatrix[:, ind_ID]
        ID_vector_list = [neuron_vector[k * num_per_class: k * num_per_class + num_per_class] for k in range(num_of_ID)]
        ID_vector_list = np.array(ID_vector_list)
        mean_list = np.mean(ID_vector_list, axis=1)
        avgMatrix_ID.append(mean_list)
    avgMatrix_ID = np.array(avgMatrix_ID).transpose()
    print(avgMatrix_ID.shape)

    avgMatrix_nonID = []
    for ind_nonID in mask_nonID:
        neuron_vector = fullMatrix[:, ind_nonID]
        ID_vector_list = [neuron_vector[k * num_per_class: k * num_per_class + num_per_class] for k in range(num_of_ID)]
        ID_vector_list = np.array(ID_vector_list)
        mean_list = np.mean(ID_vector_list, axis=1)
        avgMatrix_nonID.append(mean_list)
    avgMatrix_nonID = np.array(avgMatrix_nonID).transpose()
    print(avgMatrix_nonID.shape)
    del fullMatrix

    with open(root + '/' + layer + '_avgMatrix_Full.pkl', 'wb') as f:
        pickle.dump(avgMatrix_Full, f, protocol=4)
    with open(root + '/' + layer + '_avgMatrix_ID.pkl', 'wb') as f:
        pickle.dump(avgMatrix_ID, f, protocol=4)
    with open(root + '/' + layer + '_avgMatrix_nonID.pkl', 'wb') as f:
        pickle.dump(avgMatrix_nonID, f, protocol=4)

# === 2. Compute Euclidean Distance Matrix for each avgMatrix (50 * 50) ===
    print('Computing Euclidean Distance Matrix...')
    distanceMatrix_Full = euclidean_distances(avgMatrix_Full, avgMatrix_Full)
    distanceMatrix_ID = euclidean_distances(avgMatrix_ID, avgMatrix_ID)
    distanceMatrix_nonID = euclidean_distances(avgMatrix_nonID, avgMatrix_nonID)
    del avgMatrix_Full, avgMatrix_ID, avgMatrix_nonID

# === 3. Vectorize the Distance Matrix (1225 * 1) ===
    print('Vectorizing the Distance Matrix...')
    col_Full = distanceMatrix_Full.shape[-1]
    distanceList_Full = []
    for i in range(col_Full):
        for item in distanceMatrix_Full[i + 1:, i]:
            distanceList_Full.append(item)
    DistMat_Full.update({layer: distanceList_Full})

    col_ID = distanceMatrix_ID.shape[-1]
    distanceList_ID = []
    for i in range(col_ID):
        for item in distanceMatrix_ID[i + 1:, i]:
            distanceList_ID.append(item)
    DistMat_ID.update({layer: distanceList_ID})

    col_nonID = distanceMatrix_nonID.shape[-1]
    distanceList_nonID = []
    for i in range(col_nonID):
        for item in distanceMatrix_nonID[i + 1:, i]:
            distanceList_nonID.append(item)
    DistMat_nonID.update({layer: distanceList_nonID})

# === 4. Save to .mat (1225 * 21)===
savemat(root + '/' + 'DistMat_Full.mat', DistMat_Full)
savemat(root + '/' + 'DistMat_ID.mat', DistMat_ID)
savemat(root + '/' + 'DistMat_nonID.mat', DistMat_nonID)
print('Save complete!')
