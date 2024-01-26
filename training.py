## Complete training and testing function for your 3D Voxel GAN and have fun making pottery art!

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator, Discriminator
import click
from utils.model_utils import *
import argparse
from test import *
from utils import visualize

def CVAE_loss(z, x, mean, logstd, ratio):
    MSEcriterion = nn.MSELoss().to(available_device)
    mse = MSEcriterion(x, z)
    var = torch.pow(torch.exp(logstd), 2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return mse + kld * ratio


def main():
    # Here is a simple demonstration argparse, you may customize your own implementations, and
    # your hyperparam list MAY INCLUDE:
    # 1. Z_latent_space
    # 2. G_lr
    # 3. D_lr  (learning rate for Discriminator)
    # 4. betas if you are going to use Adam optimizer
    # 5. Resolution for input data
    # 6. Training Epochs
    # 7. Test per epoch
    # 8. Batch Size
    # 9. Dataset Dir
    # 10. Load / Save model Device
    # 11. test result save dir
    # 12. device!
    # .... (maybe there exists more hyperparams to be appointed)
    epochs = 100
    G_lr = 2e-3
    D_lr = 2e-4
    C_lr = 2e-4
    optimizer = 'ADAM'
    beta1 = 0.9
    beta2 = 0.999
    batch_size = 64  # modify according to device capability
    n_labels = 11
    resolution = 32
    z_latent_space = 64
    log_interval = 100
    vi_ratio = 1
    dis_ratio = 1
    c_ratio = 0.5

    dirdataset = "../VoxPottery"
    available_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='training/testing')
    parser.add_argument('-r', type=int, help='resolution')
    args = parser.parse_known_args()[0]

    ### Initialize train and test dataset
    dtrain = FragmentDataset(dirdataset, 'train', resolution)
    dtest = FragmentDataset(dirdataset, 'test', resolution)
    print("Data initialized")

    ### Initialize Generator and Discriminator to specific device
    G = Generator(n_labels, resolution, z_latent_space).to(available_device)
    D = Discriminator(1, resolution).to(available_device)
    C = Discriminator(n_labels, resolution).to(available_device)
    optimG = optim.Adam(G.parameters(), G_lr, (beta1, beta2))
    optimD = optim.Adam(D.parameters(), D_lr, (beta1, beta2))
    optimC = optim.Adam(C.parameters(), C_lr, (beta1, beta2))
    print("VAE initialized")

    ### Call dataloader for train and test dataset
    trainloader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=2)

    ### Implement GAN Loss!!
    # TODO
    criterion = nn.BCELoss().to(available_device)  # BCE loss
    # loss_function = 'BCE'

    ### Training Loop implementation
    ### You can refer to other papers / github repos for training a GAN
    # TODO
    print("Start training")
    for epoch in range(epochs):
        lossG = 0
        for i, data in enumerate(trainloader, 0):
            frg, vox, label = data
            vox = vox.to(available_device)
            frg = frg.to(available_device)
            frg = frg.unsqueeze(1).float()
            vox = vox.unsqueeze(1).float()
            whole = vox + frg
            label_onehot = torch.zeros((vox.shape[0], n_labels)).to(available_device)
            label_onehot[torch.arange(vox.shape[0]), label] = 1
            # train classifier on 11 types of ceramics (prepare for conditional GAN)
            # print("training C")
            out = C(whole)
            truth = label_onehot.to(available_device)
            lossC = criterion(out, truth)
            C.zero_grad()
            lossC.backward()
            optimC.step()
            # train Discriminator
            # print("training D")
            out = D(whole)
            real_label = torch.ones(batch_size).to(available_device)  # real pieces labelled 1
            fake_label = torch.zeros(batch_size).to(available_device)  # fake pieces labelled 0
            lossD_real = criterion(out.squeeze(), real_label)

            z = torch.randn(batch_size, z_latent_space + n_labels).to(available_device)
            fake_data = G.forward_decode(z) + vox
            out = D(fake_data)
            lossD_fake = criterion(out.squeeze(), fake_label)

            lossD = lossD_real + lossD_fake
            D.zero_grad()
            lossD.backward()
            optimD.step()
            # train Generator
            # print("training G")
            z, mean, logstd = G.forward_encode(vox)
            z = torch.cat([z, label_onehot], 1)
            recon_data = G.forward_decode(z)
            lossG_var_completion = CVAE_loss(recon_data, vox, mean, logstd, vi_ratio)
            out = D(recon_data + vox)
            truth = torch.ones(batch_size).to(available_device)
            lossG_dis = criterion(out.squeeze(), truth)
            out = C(recon_data + vox)
            truth = label_onehot
            lossG_condition = criterion(out.squeeze(), truth)
            G.zero_grad()
            lossG = lossG_var_completion + dis_ratio * lossG_dis + c_ratio * lossG_condition
            lossG.backward()
            optimG.step()
            # print("training loop complete")
            if i % log_interval == 0:
                print("i =", i, ", loss =", lossG)
                # test()
        if epoch % 10 == 0:
            print("EPOCH", epoch, lossG)

    print("Finish training")
    PATH = './Generator.pth'
    torch.save(G.state_dict(), PATH)
    PATH = './Discriminator.pth'
    torch.save(D.state_dict(), PATH)
    PATH = './Classifier.pth'
    torch.save(C.state_dict(), PATH)


if __name__ == "__main__":
    main()
