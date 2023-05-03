import numpy as np
import torch
import torch.nn as nn

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def weighted_RMSE(pred, true, alpha):
    diff = pred - true
    weighted_diff = np.where(diff > 0, diff*alpha, diff)
    return np.sqrt((weighted_diff**2).mean()).round(2)
    
class RevenueLoss(nn.Module):
    def __init__(self):
        super(RevenueLoss, self).__init__()
    
    def forward(self, pred, true):
        return torch.where(pred > true, 0, pred).sum()*(-1)
    
class WeightedRMSE(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, alpha):
        super(WeightedRMSE, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, true):
        diff = pred - true
        weighted_diff = torch.where(diff > 0, diff*self.alpha, diff)
        return torch.sqrt((weighted_diff**2).mean())
    
class LinExpLoss(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, linexp_weight):
        super(LinExpLoss, self).__init__()
        self.linexp_weight = linexp_weight
    
    def forward(self, pred, true):
        diff = pred - true # this order matters
        linexp_weight = torch.tensor(self.linexp_weight)
        loss = (2/torch.pow(linexp_weight, 2))*(torch.exp(linexp_weight*diff)- linexp_weight*diff - 1)
        return torch.sqrt((loss).mean())


class PositiveMSE(nn.Module):
    def __init__(self):
        super(PositiveMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        mse_loss = self.mse(pred, true)
        neg_loss = torch.mean(torch.abs(torch.min(pred, torch.zeros_like(pred))))
        return mse_loss + neg_loss
    
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        return self.mse(torch.log(pred + 1), torch.log(true + 1))

