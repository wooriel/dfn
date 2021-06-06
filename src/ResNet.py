import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, train, in_channels, hid_channels, out_channels):
        super(ResBlock, self).__init__()
        '''
        Args:
            train: train/test
            in_channels: input descriptor per point (batch * num_pts * #channels)
            # hid_channels.....I just made it
            # out_channels: output descriptor per point (same as in_channels) <- I just removed
        '''
        
        self.fc = nn.linear(in_channels, hid_channels, bias=False)
        self.bn = nn.BatchNorm1d(hid_channels, eps=1e-3, momentum=1e-3)
        self.relu = nn.ReLU()
        self.fc2 = nn.linear(hid_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=1e-3)
        self.fc3 = nn.linear(in_channels, out_channels, bias=False)

        
    def forward(self, x):
        identity = x
        output = self.relu(self.bn(self.fc(x)))
        output = self.bn2(self.fc2(output))
        
        # dim change
        if not torch.equal(identity, output):
            identity = self.fc3(identity)
            
        output += identity
        output = self.relu(output)
        
        return output
        
class DFM(nn.Module):
    def __init__(self, train):
        super(DFM, self).__init__()
        """
        src_eigen, tar_eigen, trans_src_eigen, trans_tar_eigen, dist_map
        Args:
            train: train/test
            src_shot: SHOT of 1st model shape
            tar_shot: SHOT of 2nd model shape
            src_eigen: eigen decomposition of 1st model shape
            tar_eigen: eigen decomposition of 2nd model shape
            trans_src_eigen: transpose matrix of src_eigen
            trans_tar_eigen: transpose matrix of tar_eigen
            dist_map: distance map on target shape
        """
        self.train = train
        self.res_layer = ResBlock(train, 352, 352, 352)
        
    def forward(self, src_desc, tar_desc, src_eigen, tar_eigen):
        for i in range(num_layer):
            src_desc = ResBlock(src_desc)
            tar_desc = ResBlock(tar_desc)
        # project to eigen vecs
        F_mat = torch.bmm(src_eigen.transpose(2, 1), src_desc)
        G_mat = torch.bmm(tar_eigen.transpose(2, 1), tar_desc)
        F_trans = F_mat.transpose(2, 1) # Batch_size*120*352
        G_trans = G_mat.transpose(2, 1)
        C_mat = torch.empty(src_eigen.size(2), src_eigen.size(2)).unsqueeze(dim=0)
        for i in range(src_desc.size(0)):
            inv = torch.linalg.inv(torch.matmul(F_mat[i], F_trans[i]))
            C = torch.matmul(torch.matmul(inv, F_trans), G_mat[i]) # 120*120
            if i == 0:
                C_mat = C.unsqueeze(0)
            else:
                C_mat = torch.cat((C_mat, C.unsqueeze(0)), dim=0)
        C = C.transpose(2, 1)
        # soft correspondence matrix
        P = torch.abs(torch.bmm(torch.bmm(tar_eigen, C), src_eigen.transpose(2, 1)))
        P = F.normalize(P, p=2, dim=1)
        
        return P, C