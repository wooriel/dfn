import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
    
    def forward(self, P, dist_x, dist_y):
        criterion = nn.MSELoss(reduce=True)
        loss = criterion(dist_x, torch.bmm(torch.transpose(2, 1), torch.bmm(dist_y, Q)))
        return loss