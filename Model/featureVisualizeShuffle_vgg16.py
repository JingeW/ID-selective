import os
import pickle
import skimage.transform
import matplotlib.image

# pic_path = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_CelebA_original_featureMaps_kernel_shuffle/40/000030'
pic_path = '/home/sdb1/Jinge/ID_selective/VGG16_Vggface_CelebA_original_featureMaps_layer_shuffle/40/000030'

# dst = '/home/sdb1/Jinge/ID_selective/Result/Visualize/VGG16_Vggface_CelebA_original_kernel_shuffle/'
dst = '/home/sdb1/Jinge/ID_selective/Result/Visualize/VGG16_Vggface_CelebA_original_layer_shuffle/'
if not os.path.exists(dst):
    os.makedirs(dst)

layer_list = sorted([os.path.join(pic_path, folder) for folder in os.listdir(pic_path)])
for layer in layer_list:
    item_list = sorted([os.path.join(layer, folder) for folder in os.listdir(layer)])
    tokens = layer.split('/')
    tokens = tokens[-3:]
    save_dir = dst + '/'.join(tokens)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for item in item_list:
        with open(item, 'rb') as f:
            featMap = pickle.load(f)
        if not featMap.shape[0] == 224:
            featMap = skimage.transform.resize(featMap, (224, 224))
        name = item.split('/')[-1].split('.')[0] + '.png'
        matplotlib.image.imsave(save_dir + '/' + name, featMap)
