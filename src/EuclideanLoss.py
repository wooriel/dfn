import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
    
    def forward(self, P, dist_x, dist_y):
        loss = torch.sqrt(((P * dist_y) ** 2).sum((1, 2)))
#         criterion = nn.MSELoss(reduce=True)
#         loss = criterion(dist_x, torch.bmm(torch.transpose(P, 2, 1), torch.bmm(dist_y, P)))
        return torch.mean(loss)