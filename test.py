## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator, Discriminator
from utils.visualize import plot_join

def metric(out, frg):
    A = 0
    B = 0
    S = 0
    for x in range(frg.shape[0]):
        for y in range(frg.shape[1]):
            for z in range(frg.shape[2]):
                if out[x, y, z] != 0:
                    A += 1
                elif frg[x, y, z] != 0:
                    S += 1
                if frg[x, y, z] != 0:
                    B += 1
    S += A
    return 2 * S / (A + B)

def test():
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.   

    # create testing dataset
    n_labels = 11
    resolution = 32
    z_latent_space = 64
    metrics = ['DSC', 'JD', 'MSE']
    batch_size = 64  # modify according to device capability
    available_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G = Generator(n_labels, resolution, z_latent_space).to(available_device)
    dirdataset = "../VoxPottery"
    save_dir = "../visualization"
    dtest = FragmentDataset(dirdataset, 'test', resolution)
    testloader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    # test
        # forward
    with torch.no_grad():
        for data in testloader:
            frg, vox, label = data
            vox = vox.to(available_device)
            frg = frg.to(available_device)

            out = G.forward(vox)
            dis = metric(out, frg)
            correct += dis
            total += 1
            plot_join(vox, out)
        rate = 100 * correct // total
        print(f'Accuracy of the network: {rate} %')

    #return