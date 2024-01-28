## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
import utils
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator
from utils.visualize import plot_join, plot
from utils import pyvox
from utils.pyvox import writer
from utils.pyvox.models import Vox
n_labels = 11
resolution = 32
z_latent_space = 64
batch_size = 16

def DSC(out, frg):
    dis = np.zeros((frg.shape[0]))
    #print("DSC")
    for i in range(frg.shape[0]):
        A = 0
        B = 0
        S = 0
        eps = 1e-4
        for x in range(frg.shape[2]):
            for y in range(frg.shape[3]):
                for z in range(frg.shape[4]):
                    if out[i, 0, x, y, z] > eps:
                        A += 1
                    elif frg[i, 0, x, y, z] > eps:
                        S += 1
                    if frg[i, 0, x, y, z] > eps:
                        B += 1
        S += A
        dis[i] = S / (A + B)
    #print("over")
    return np.mean(dis)


def JD(out, frg):
    dis = np.zeros((frg.shape[0]))
    eps = 1e-4
    for i in range(frg.shape[0]):
        U = 0
        N = 0
        for x in range(frg.shape[2]):
            for y in range(frg.shape[3]):
                for z in range(frg.shape[4]):
                    if out[i, 0, x, y, z] > eps or frg[i, 0, x, y, z] > eps:
                        U += 1
                    if out[i, 0, x, y, z] > eps and frg[i, 0, x, y, z] > eps:
                        N += 1
        dis[i] = 1 - (U - N) / U
    return np.mean(dis)


def test(metric):
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.   

    # create testing dataset
    available_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dirdataset = "/content/drive/MyDrive/Colab Notebooks/data_voxelized"
    save_dir = "/content/drive/MyDrive/Colab Notebooks/utils/voxfile/"
    vis_dir = "/content/drive/MyDrive/Colab Notebooks/utils/visualization/"
    dtest = FragmentDataset(dirdataset, 'test', resolution)
    testloader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=2)
    G = Generator(n_labels, resolution, z_latent_space).to(available_device)
    G.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Generator.pth'))
    G.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        print('start testing')
        i = 1
        for data in testloader:
            frg, vox, label = data
            vox = vox.to(available_device).unsqueeze(1).float()
            frg = frg.to(available_device).unsqueeze(1).float()
            label_onehot = torch.zeros((vox.shape[0], n_labels)).to(available_device)
            label_onehot[torch.arange(vox.shape[0]), label] = 1
            z, mean, logstd = G.forward_encode(vox)
            z = torch.cat([z, label_onehot], 1)
            out = G.forward_decode(z)
            #a = out.cpu().detach().numpy()[0][0]
            #file = Vox.from_dense(a)
            #pyvox.writer.VoxWriter(save_dir + str(i) + '.vox', file).write()
            dis = 0
            if metric == 'DSC':
                dis = DSC(out, frg)
            elif metric == 'JD':
                dis = JD(out, frg)
            print(i, dis)
            correct += dis
            total += 1
            #np.savetxt(save_dir+str(i) + ".txt", out.cpu().detach().numpy()[0][0])
            plot(out.cpu().detach().numpy()[0][0], vis_dir+str(i)+'.jpg')
            i += 1
            #plot_join(vox.cpu().detach().numpy()[0][0], out.cpu().detach().numpy()[0][0])
        rate = 100 * correct // total
        print("accuracy", rate)
        return rate