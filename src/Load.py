import os
import h5py
import numpy as np
import torch
import torch.utils.data as data

data_dir = '../data/'
train_reg_shot = 'training/reg_100/'
train_scan_shot = 'training/scan_shot/'
train_eigen = 'training/reg_lb/'
train_dist = 'training/l2_dist/'
test_scan_shot = 'test/scan_shot/'
test_eigen = 'test/scan_lb/'
inter = 'inter_challenge.txt'
intra = 'intra_challenge.txt'


def load_pairs():
        intraFname = ''.join([data_dir, intra])
        interFname = ''.join([data_dir, inter])
        pairs = []
        with open(intraFname) as f:
            for line in f:
                array = line.split('_')
                array[1] = array[1][:-1]
                pairs.append(array)
        with open(interFname) as f:
            for line in f:
                array = line.split('_')
                array[1] = array[1][:-1]
                pairs.append(array)
        return pairs
        
def load_shot(fname):
    i = 0
    shot = []
    if fname == '':
        return
    with open(fname, 'r') as f:
        for line in f:
            array = line.split()
            if len(array) > 0:
                np_array = np.array(array)
                f_array = np_array.astype(float)
                t_array = torch.from_numpy(f_array)
                shot.append(f_array)
            i = i + 1
    shot_np = np.array(shot)
    shot_np = shot_np[shot_np[:,0].argsort()]
    shot_ret = shot_np[:, 4:]
    shot_ret = shot_np
    t_shot = torch.from_numpy(shot_ret) #shot_ret
#         t_shot = torch.unsqueeze(t_shot, 0).float().to(cuda_device)
    f.close()
    return t_shot
    
def load_eig(fname):
    file = h5py.File(fname, 'r')
    dkey = list(file.keys())[0]
    dset = file[dkey]
    nData = np.array(dset)
    nData = np.transpose(nData)
    numVert = int(nData[0][0])
    numEig = int(nData[0][1])
    phi = nData[1]

    vertice = nData[2:numVert+2]
    nd_vert = np.array(vertice)
    nd_vert = nd_vert.astype(float)
    t_vert = torch.from_numpy(nd_vert)
    file.close()
    return t_vert

# def load_dist(fname):
#     file = h5py.File(fname, 'r')
#     dkey = list(file.keys())[0]
#     dset = file[dkey]
#     nData = np.array(dset)
#     nData = np.transpose(nData)

#     vertice = nData[:]
#     nd_vert = np.array(vertice)
# #     nd_vert = nd_vert.astype(float)
#     t_vert = torch.from_numpy(nd_vert)
#     file.close()
#     return t_vert