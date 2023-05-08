import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms, utils, datasets
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0),  # (8, 30, 30)
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),  # (8, 28, 28)
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),  # (8, 30, 30)
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=0),  # (8, 32, 32)
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


def show_image(img_tensor, filename):
    ToPIL = transforms.ToPILImage()
    img = ToPIL(img_tensor)
    plt.imshow(img)
    plt.savefig(filename)
    # plt.show()
    plt.close()


def test(dataset_loader, model):
    True_Image = []
    Pred_Image = []
    Noisy_Image = []
    Loss = []
    model.eval()

    with torch.no_grad():
        for j, data in enumerate(dataset_loader):
            image = data[0]
            image = image.to(device)
            Output = model(image).squeeze().cpu()
            # print(Output.shape)

            image = image.squeeze().cpu()
            # show_image(image)
            # show_image(Output)
            loss = criterion(image, Output)
            Pred_Image.append(Output)
            True_Image.append(image)
            Loss.append(loss)

    return True_Image, Pred_Image, Loss


def train(trainset_loader, model, loss_fn, opt):
    for step, data in enumerate(trainset_loader):
        model.train(mode=True)
        image = data[0]
        image = image.to(device)

        opt.zero_grad()
        Output = model(image).squeeze().cpu()
        image = image.squeeze().cpu()

        loss = criterion(image, Output)

        # Backward
        loss.backward()
        opt.step()


if __name__ == "__main__":
    # Download dataset
    tf = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.cifar.CIFAR10(root='./CIFAR10/train', train=True, transform=tf, download=False)
    test_dataset = datasets.cifar.CIFAR10(root='./CIFAR10/test', train=False, transform=tf, download=False)



    # Parameter setting
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 1e-6
    epoches = 100
    dir = './Output/Q1/'

    # Generate DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_loss = []
    noise_list = []
    test_size = 10000
    T, P, L = test(test_loader, model)
    total_loss.append(np.mean(L))
    image_num = random.sample(range(test_size), 1)
    path_list = []
    for i in image_num:
        newdir = 'Image' + str(i) + '_lr=' + str(learning_rate)
        path = os.path.join(dir, newdir)
        path_list.append(path)
        os.mkdir(path)
        print(path)
        show_image(T[i], os.path.join(path, 'True_Image.png'))
        x = range(len(L))
        plt.plot(x, L)
        plt.xlabel('Testing Samples')
        plt.ylabel('Loss')
        plt.title('Loss in Epoch0')
        plt.savefig(os.path.join(path, 'Loss_Epoch0.png'))
        plt.close()
    # plt.show()

    for epoch in range(epoches):
        print('Epoch: ', epoch + 1)
        train(train_loader, model, criterion, optimizer)
        T, P, L = test(test_loader, model)
        k = 0
        total_loss.append(np.mean(L))
        print('Total_loss: ', total_loss[-1])
        if total_loss[-1] > total_loss[-2]:
            print('Total Loss increases in epoch ', epoch)
        x = range(len(L))
        for i in image_num:
            show_image(P[i], os.path.join(path_list[k], 'Predicted_Epoch0' + str(epoch + 1) + '.png'))
            plt.plot(x, L)
            plt.xlabel('Testing Samples')
            plt.ylabel('Loss')
            plt.title('Loss in Epoch' + str(epoch + 1))
            plt.savefig(os.path.join(path_list[k], 'Loss_Epoch' + str(epoch + 1) + '.png'))
            plt.close()
            k += 1

    y = range(len(total_loss))
    k = 0
    for i in image_num:
        plt.plot(y, total_loss)
        plt.xlabel('Epoch')
        plt.ylabel('TotalLoss')
        plt.title('Curve of Total Loss and Epoch Times')
        plt.savefig(os.path.join(path_list[k], 'TotalLoss.png'))
        plt.close()
        k += 1
