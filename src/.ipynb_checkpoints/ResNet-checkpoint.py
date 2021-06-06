import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, tr, in_channels=352, hid_channels=352, out_channels=352):
        super(ResBlock, self).__init__()
        '''
        Args:
            tr: tr/test
            in_channels: input descriptor per point (batch * num_pts * #channels)
            # hid_channels.....I just made it
            # out_channels: output descriptor per point (same as in_channels) <- I just removed
        '''
        
        self.fc = nn.Linear(in_channels, hid_channels, bias=False)
        self.bn = nn.BatchNorm1d(hid_channels, eps=1e-3, momentum=1e-3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=1e-3)
        self.fc3 = nn.Linear(in_channels, out_channels, bias=False)

        
    def forward(self, x):
        identity = x
        # batchnorm1d gets (N, C, L) -> C number of features

        output = self.relu(torch.transpose(self.bn(torch.transpose(self.fc(x), 2, 1)), 2, 1))
        output = torch.transpose(self.bn2(torch.transpose(self.fc2(output), 2, 1)), 2, 1)
        
        # dim change
        if not torch.equal(identity, output):
            identity = self.fc3(identity)
            
        output += identity
        output = self.relu(output)
        
        return output
        
class DFM(nn.Module):
    def __init__(self, tr, num_layer=7):
        super(DFM, self).__init__()
        """
        src_eigen, tar_eigen, trans_src_eigen, trans_tar_eigen, dist_map
        Args:
            tr: tr/test
            src_shot: SHOT of 1st model shape
            tar_shot: SHOT of 2nd model shape
            src_eigen: eigen decomposition of 1st model shape
            tar_eigen: eigen decomposition of 2nd model shape
            trans_src_eigen: transpose matrix of src_eigen
            trans_tar_eigen: transpose matrix of tar_eigen
            dist_map: distance map on target shape
        """
        self.tr = tr
        self.num_layer = num_layer
        self.res_layer = ResBlock(self.tr, 352, 352, 352)
        
    def forward(self, src_desc, tar_desc, src_eigen, tar_eigen):
        # src_desc 16*6890*352
        # src_eigen 16*6890*352
        for i in range(self.num_layer):
            src_desc = self.res_layer(src_desc)
            tar_desc = self.res_layer(tar_desc)
        # project to eigen vecs
        F_mat = torch.bmm(torch.transpose(src_eigen, 2, 1), src_desc) # 120*352
        G_mat = torch.bmm(torch.transpose(tar_eigen, 2, 1), tar_desc)
        
        F_trans = F_mat.transpose(2, 1) # Batch_size*352*120
        G_trans = G_mat.transpose(2, 1)
        C_mats = []
        for i in range(src_desc.size(0)):
#             inv = torch.inverse(F_trans[i] @ F_mat[i]) 
            C = torch.inverse(F_trans[i] @ F_mat[i]) @ F_trans[i] @ G_mat[i] #352*352
            C_mats.append(C[:120, :120].t().unsqueeze(0))
#             C = torch.matmul(torch.matmul(inv, F_trans[i]), G_mat[i]) # 352*352
#             if i == 0:
#                 C_mat = C.unsqueeze(0)[:, 0:120, :]
#             else:
#                 C_mat = torch.cat((C_mat, C.unsqueeze(0)[:, 0:120, :]), dim=0)
#         C = C.transpose(2, 1)
        C = torch.cat(C_mats, dim=0)
        # soft correspondence matrix
        P = torch.abs(torch.bmm(torch.bmm(tar_eigen, C), src_eigen.transpose(2, 1)))
        P = F.normalize(P, p=2, dim=1)
        
        return P, C