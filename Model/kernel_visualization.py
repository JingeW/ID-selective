import os
import torchfile
import matplotlib.image
import skimage.transform

path = '/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'
dst_path = '/home/sdb1/Jinge/ID_selective/Result/Visualize_kernel'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
model = torchfile.load(path)

block = 1
counter = 1
block_size = [2, 2, 3, 3, 3]
layer1 = model.modules[0]
for i, layer in enumerate(model.modules):
    if layer.weight is not None:
        if block <= 1:
            layer_name = 'conv_%d_%d' % (block, counter)
            print(layer_name)
            counter += 1
            if counter > block_size[block - 1]:
                counter = 1
                block += 1
            channel_num, weight_num = layer.weight.shape
            for j in range(channel_num):
                for k in range(0, weight_num, 9):
                    kernel = layer.weight[j, :][k: k + 9].reshape((3, 3))
                    kernel = skimage.transform.resize(kernel, (64, 64))
                    dst_layer = os.path.join(dst_path, layer_name)
                    if not os.path.exists(dst_layer):
                        os.makedirs(dst_layer)
                    dst_kernel = os.path.join(dst_layer, str(j).zfill(2) + '_' + str(int(k / 9) + 1) + '.png')
                    matplotlib.image.imsave(dst_kernel, kernel)
