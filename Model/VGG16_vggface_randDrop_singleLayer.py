import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import numpy as np
# import pickle
# import skimage.data
# import skimage.io
# import skimage.transform
# import torchvision.transforms as transforms


class VGG_16_perturb_singleLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        # self.fc8 = nn.Linear(4096, 50)

    def load_weights(self, path='/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    # print(self_layer.weight.data.shape)
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        # load premade drop masks
        path = '/home/sdb1/Jinge/ID_selective/Model/leison_mask/masks/randDrop_0.3.npy'
        with open(path, 'rb') as f:
            masks = np.load(f, allow_pickle=True)
        # conv1
        x01 = F.relu(self.conv_1_1(x))  # x01 has the shape of ([b, c, w, h])
        # mask1_1 = torch.from_numpy(masks[0]).cuda()
        # x01 = x01 * mask1_1

        x02 = F.relu(self.conv_1_2(x01))
        # mask1_2 = torch.from_numpy(masks[1]).cuda()
        # x02 = x02 * mask1_2

        x03 = F.max_pool2d(x02, 2, 2)

        # conv2
        x04 = F.relu(self.conv_2_1(x03))
        # mask2_1 = torch.from_numpy(masks[2]).cuda()
        # x04 = x04 * mask2_1

        x05 = F.relu(self.conv_2_2(x04))
        # mask2_2 = torch.from_numpy(masks[3]).cuda()
        # x05 = x05 * mask2_2

        x06 = F.max_pool2d(x05, 2, 2)

        # conv3
        x07 = F.relu(self.conv_3_1(x06))
        # mask3_1 = torch.from_numpy(masks[4]).cuda()
        # x07 = x07 * mask3_1

        x08 = F.relu(self.conv_3_2(x07))
        # mask3_2 = torch.from_numpy(masks[5]).cuda()
        # x08 = x08 * mask3_2

        x09 = F.relu(self.conv_3_3(x08))
        # mask3_3 = torch.from_numpy(masks[6]).cuda()
        # x09 = x09 * mask3_3

        x10 = F.max_pool2d(x09, 2, 2)

        # conv4
        x11 = F.relu(self.conv_4_1(x10))
        # mask4_1 = torch.from_numpy(masks[7]).cuda()
        # x11 = x11 * mask4_1

        x12 = F.relu(self.conv_4_2(x11))
        # mask4_2 = torch.from_numpy(masks[8]).cuda()
        # x12 = x12 * mask4_2

        x13 = F.relu(self.conv_4_3(x12))
        # mask4_3 = torch.from_numpy(masks[9]).cuda()
        # x13 = x13 * mask4_3

        x14 = F.max_pool2d(x13, 2, 2)

        # conv5
        x15 = F.relu(self.conv_5_1(x14))
        # mask5_1 = torch.from_numpy(masks[10]).cuda()
        # x15 = x15 * mask5_1

        x16 = F.relu(self.conv_5_2(x15))
        # mask5_2 = torch.from_numpy(masks[11]).cuda()
        # x16 = x16 * mask5_2

        x17 = F.relu(self.conv_5_3(x16))
        mask5_3 = torch.from_numpy(masks[12]).cuda()
        x17 = x17 * mask5_3

        x18 = F.max_pool2d(x17, 2, 2)

        # FC
        x19 = x18.view(x18.size(0), -1)
        x20 = F.relu(self.fc6(x19))
        x21 = F.dropout(x20, 0.5, self.training)
        x22 = F.relu(self.fc7(x21))
        x23 = F.dropout(x22, 0.5, self.training)

        x24 = F.relu(self.fc8(x23))
        feat = [
            x01, x02, x03,
            x04, x05, x06,
            x07, x08, x09, x10,
            x11, x12, x13, x14,
            x15, x16, x17, x18,
            x20, x22, x24
        ]

        return feat

        # return self.fc8(x23)
        # return F.softmax(self.fc8(x23), dim=1)


# if __name__ == "__main__":
#     test_loader = torch.load('/home/sdb1/Jinge/ID_selective/Model/pth/test_loader.pth')
#     drop_model = VGG_16_perturb()
#     drop_model.load_state_dict(torch.load('/home/sdb1/Jinge/ID_selective/Model/pth/9.pth'))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     drop_model.to(device)
#     criterion = nn.CrossEntropyLoss()

#     drop_test_loss = 0
#     drop_acc = 0
#     drop_model.eval()
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             drop_res = drop_model(inputs)
#             drop_batch_loss = criterion(drop_res, labels)

#             drop_test_loss += drop_batch_loss.item()

#             drop_pred = drop_res.data.max(1, keepdim=True)[1]
#             drop_acc += torch.sum(drop_pred == labels.data).item()

#     print(
#         f"Drop_Test loss: {drop_test_loss/len(test_loader):.3f}",
#         f"Drop_Test accuracy: {drop_acc/len(test_loader):.3f}"
#     )
