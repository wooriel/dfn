import os
import h5py
import numpy as np
import torch
import torch.utils.data as data

# from Load import load_pairs, load_shot, load_eig, load_dist

data_dir = '../data/'
train_reg_shot = 'training/reg_shot/'
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
    with open(intraFname, 'r') as f:
        for line in f:
            array = line.split('_')
            array[1] = array[1][:-1]
            pairs.append(array)
    with open(interFname, 'r') as f:
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

    return t_vert

def load_dist(fname):
    file = h5py.File(fname, 'r')
    dkey = list(file.keys())[0]
    dset = file[dkey]
    nData = np.array(dset)
    nData = np.transpose(nData)

    vertice = nData[:]
    nd_vert = np.array(vertice)
    nd_vert = nd_vert.astype(float)
    t_vert = torch.from_numpy(nd_vert)

    return t_vert


class FAUSTDataset(data.Dataset):
    def __init__(self, tr, batch_length=16, eignum=120):
        self.tr = tr
        self.shot_des = []
        self.eigen_des = []
        self.dist_map = []
        self.batch_length=batch_length
        self.eignum = eignum
        
        if self.tr:
            streig = "%d" % (eignum)
            for i in range(self.batch_length):
                strnum = '%03d' % (i)
                fn = ''.join([data_dir, train_reg_shot, 'tr_reg_res_', strnum, '.txt'])
                self.shot_des.append(load_shot(fn))
                fn_eig = ''.join([data_dir, train_eigen, 'tr_reg_', streig, '_', strnum, '.h5'])
                self.eigen_des.append(load_eig(fn_eig))
                fn_dist = ''.join([data_dir, train_dist, 'tr_reg_dist_', strnum, '.h5'])
                self.dist_map.append(load_dist(fn_dist))
        else:
            pairs = load_pairs()
            for i in range(16):  
                fn1 = ''.join([data_dir, test_scan_shot, 'test_scan_d_res_', pairs[i][0], '.txt'])
                fn2 = ''.join([data_dir, test_scan_shot, 'test_scan_d_res_', pairs[i][1], '.txt'])
                self.shot_des.append(load_shot(fn1))
                self.shot_des.append(load_shot(fn2))
                fn_eig1 = ''.join([data_dir, test_eigen, 'test_scan_', streig, '_', pairs[i][0], '.h5'])
                fn_eig2 = ''.join([data_dir, test_eigen, 'test_scan_', streig, '_', pairs[i][1], '.h5'])
                self.eigen_des.append(load_eig(fn_eig1))
                self.eigen_des.append(load_eig(fn_eig2))
    def __len__(self):
        if self.tr:
            return len(self.shot_des)
        else:
            return len(self.shot_des)//2
        
    def __getitem__(self, idx):
        if self.tr:
            # I will pass idx directly to src
            # target sub will be random that does not overlaps with src
#             categories = [i for i in range(10)]
#             t_idx = np.random.choice(categories)*10+np.random.choice(categories)
            random = [i for i in range(self.batch_length)]
            t_idx = np.random.choice(random)
            while t_idx == idx:
                t_idx = np.random.choice(random)
#                 t_idx = np.random.choice(categories)*10+np.random.choice(categories)
            s_src = self.shot_des[idx]
            s_tar = self.shot_des[t_idx]
            e_src = self.eigen_des[idx]
            e_tar = self.eigen_des[t_idx]
            d_src = self.dist_map[idx]
            d_tar = self.dist_map[t_idx]
#                 stridx = "%03d" % (idx)
#                 str_tidx = "%03d" % (t_idx)
#                 fn_dist = ''.join([data_dir, train_dist, 'tr_reg_dist_', stridx, '.h5'])
#                 fn_tdist = ''.join([data_dir, train_dist, 'tr_reg_dist_', str_tidx, '.h5'])
#                 d_src = load_dist(fn_dist)
#                 d_tar = load_dist(fn_tdist)
        else:
            pairs = load_pairs()
            src_idx = int(pairs[idx][0])
            tar_idx = int(pairs[idx][1])
            s_src = self.shot_des[src_idx]
            s_tar = self.shot_des[tar_idx]
            e_src = self.eigen_des[src_idx]
            e_tar = self.eigen_des[tar_idx]
            d_src = 0
            d_tar = 0
        return s_src, s_tar, e_src, e_tar, d_src, d_tar
        
