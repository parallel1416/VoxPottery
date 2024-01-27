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


def DSC(out, frg):
    dis = np.zeros((frg.shape[0]))
    for i in range(frg.shape[0]):
        A = 0
        B = 0
        S = 0
        eps = 1e-4
        for x in range(frg.shape[2]):
            for y in range(frg.shape[3]):
                for z in range(frg.shape[4]):
                    if out[i, 0, x, y, z] < eps:
                        A += 1
                    elif frg[i, 0, x, y, z] < eps:
                        S += 1
                    if frg[i, 0, x, y, z] < eps:
                        B += 1
        S += A
        dis[i] = S / (A + B)
    return np.mean(dis)


def JD(out, frg):
    U = 0
    N = 0
    for x in range(frg.shape[0]):
        for y in range(frg.shape[1]):
            for z in range(frg.shape[2]):
                if out[x, y, z] != 0 or frg[x, y, z] != 0:
                    U += 1
                if out[x, y, z] != 0 and frg[x, y, z] == out[x, y, z]:
                    N += 1

    return 1 - (U - N) / U


def test(resolution, batch_size, metric, G):
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.   

    # create testing dataset
    available_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dirdataset = "../VoxPottery"
    save_dir = "../visualization"
    dtest = FragmentDataset(dirdataset, 'test', resolution)
    testloader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            frg, vox, label = data
            vox = vox.to(available_device)
            frg = frg.to(available_device)

            out = G.forward(vox)
            if metric == 'DSC':
                dis = DSC(out, frg)
            elif metric == 'JD':
                dis = JD(out, frg)
            correct += dis
            total += 1
            plot_join(vox, out)
        rate = 100 * correct // total
        return rate
