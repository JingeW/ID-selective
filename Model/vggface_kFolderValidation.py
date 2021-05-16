# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# import matplotlib.pyplot as plt
import torchfile
import os
from sklearn.model_selection import StratifiedKFold


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
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

    def load_weights(self, path='/home/sdb1/Jinge/ID_selective/Model/VGG_FACE.t7'):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
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
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    if not block == 8:
                        self_layer = getattr(self, "fc%d" % (block))
                        block += 1
                        self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                        self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x01 = F.relu(self.conv_1_1(x))
        x02 = F.relu(self.conv_1_2(x01))
        x03 = F.max_pool2d(x02, 2, 2)
        x04 = F.relu(self.conv_2_1(x03))
        x05 = F.relu(self.conv_2_2(x04))
        x06 = F.max_pool2d(x05, 2, 2)
        x07 = F.relu(self.conv_3_1(x06))
        x08 = F.relu(self.conv_3_2(x07))
        x09 = F.relu(self.conv_3_3(x08))
        x10 = F.max_pool2d(x09, 2, 2)
        x11 = F.relu(self.conv_4_1(x10))
        x12 = F.relu(self.conv_4_2(x11))
        x13 = F.relu(self.conv_4_3(x12))
        x14 = F.max_pool2d(x13, 2, 2)
        x15 = F.relu(self.conv_5_1(x14))
        x16 = F.relu(self.conv_5_2(x15))
        x17 = F.relu(self.conv_5_3(x16))
        x18 = F.max_pool2d(x17, 2, 2)
        x19 = x18.view(x18.size(0), -1)
        x20 = F.relu(self.fc6(x19))
        x21 = F.dropout(x20, 0.5, self.training)
        x22 = F.relu(self.fc7(x21))
        x23 = F.dropout(x22, 0.5, self.training)
        # return self.fc8(x23)
        return F.softmax(self.fc8(x23), dim=1)


def load_split_train_test(root, valid_size):

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root, transform=data_transform)
    test_data = datasets.ImageFolder(root, transform=data_transform)

    num_train = len(train_data)         # num of training set, actually ATM this is the total num of img under root
    indices = list(range(num_train))    # creat indices for randomize

    split = int(np.floor(valid_size * num_train))  # pick the break point based on the ratio given by validtion size
    np.random.shuffle(indices)          # randomize the data indices

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)  # randomize again
    test_sampler = SubsetRandomSampler(test_idx)

    # ===================load data========================
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=4)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
    return train_loader, test_loader


if __name__ == "__main__":
    root = '/home/sdb1/Jinge/ID_selective/Data/CelebA_original'

    acc_list = []
    for i in range(10):
        train_loader, test_loader = load_split_train_test(root, 0.33)
        # print(train_loader.dataset.classes)
        # torch.save(test_loader, '/home/sdb1/Jinge/ID_selective/Model/pth/test_loader.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VGG_16()
        model.load_weights()
        for param in model.parameters():
            param.requires_grad = False

        model.fc8 = nn.Linear(4096, 50)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        learning_rata = 1e-3 * 5
        my_optimizer = optim.Adam(model.parameters(), lr=learning_rata)
        my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.9)
        epochs = 10
        step = 0
        running_loss = 0
        train_losses, test_losses = [], []
        best_acc = 0
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)
                # print('input shape', inputs.shape)
                my_optimizer.zero_grad()
                out = model(inputs)
                # print(out)
                # loss = criterion(out, labels.unsqueeze(1))
                loss = criterion(out, labels)

                loss.backward()
                my_optimizer.step()
                running_loss += loss.item()
                step += 1

                if(step + 1) % 5 == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            res = model(inputs)
                            # print(res)
                            # batch_loss = criterion(res, labels.unsqueeze(1))
                            batch_loss = criterion(res, labels)

                            test_loss += batch_loss.item()

                            pred = res.data.max(1, keepdim=True)[1]
                            accuracy += torch.sum(pred == labels.data).item()
                    train_losses.append(running_loss / len(train_loader))
                    test_losses.append(test_loss / len(test_loader))
                    test_acc = accuracy / len(test_loader)
                    if best_acc < test_acc:
                        best_acc = test_acc

                    print(
                        f"Epoch {str(epoch+1).zfill(2)}/{epochs}",
                        f"Train loss: {running_loss/5:.3f}",
                        f"Test loss: {test_loss/len(test_loader):.3f}",
                        f"Test accuracy: {test_acc:.3f}",
                        f"Best acc:{best_acc}"
                    )
                    running_loss = 0
            pth_dir = "/home/sdb1/Jinge/ID_selective/Model/pth"
            if not os.path.exists(pth_dir):
                os.makedirs(pth_dir)
            pth_path = os.path.join(pth_dir, str(epoch) + '.pth')
            torch.save(model.state_dict(), pth_path)
            print('model saved!')
            my_lr_scheduler.step()
        acc_list.append(best_acc)
    # print("Accuracy 95 CI: %0.3f (+/- %0.3f)" % (acc_list.mean(), acc_list.std() * 2))
    # plt.plot(train_losses, label="Training loss")
    # plt.plot(test_losses, label='Validation loss')
    # plt.legend(frameon=False)
    # plt.savefig('./loss.png')
    # plt.show()
