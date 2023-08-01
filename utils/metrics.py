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
    return np.mean((pred-true)**2).round(2)

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

def weighted_RMSE(pred, true, wrmse_weight):
    diff = pred - true
    weighted_diff = np.where(diff > 0, diff*wrmse_weight, diff)
    return np.sqrt((weighted_diff**2).mean()).round(2)

def LinEx(pred, true, linex_weight):
    diff = pred - true # this order matters
    loss = (2/np.power(linex_weight, 2))*(np.exp(linex_weight*diff)- linex_weight*diff - 1)
    return np.sqrt((loss).mean()).round(2)

def LinLin(pred, true, linlin_weight):
    diff = true - pred # positive = underestimation, negative = overestimation
    loss = np.where(diff < 0, -diff*linlin_weight, diff*(1-linlin_weight))
    return loss.mean().round(2)
    
class RevenueLoss(nn.Module):
    def __init__(self):
        super(RevenueLoss, self).__init__()
    
    def forward(self, pred, true):
        return torch.where(pred > true, 0, pred).sum()*(-1)
    
class WeightedRMSE(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, wrmse_weight):
        super(WeightedRMSE, self).__init__()
        self.wrmse_weight = wrmse_weight
    
    def forward(self, pred, true):
        diff = pred - true
        weighted_diff = torch.where(diff > 0, diff*self.wrmse_weight, diff)
        return torch.sqrt((weighted_diff**2).mean())
    
class LinLinLoss(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, linlin_weight):
        super(LinLinLoss, self).__init__()
        self.linlin_weight = linlin_weight # prob of underforcast, 1- : overforecast
    
    def forward(self, pred, true):
        diff = pred - true
        weighted_diff = torch.where(diff < 0, -diff*self.linlin_weight, diff*(1-self.linlin_weight))
        return weighted_diff.mean() # MAE?
     
class LinExLoss(nn.Module):
    def __init__(self, linex_weight):
        super(LinExLoss, self).__init__()
        self.linex_weight = linex_weight
    
    def forward(self, pred, true):
        diff = pred - true # this order matters
        linex_weight = torch.tensor(self.linex_weight)
        a = (2/torch.pow(linex_weight, 2))
        b = (torch.exp(linex_weight*diff)- linex_weight*diff - 1)
        loss = (2/torch.pow(linex_weight, 2))*(torch.exp(linex_weight*diff)- linex_weight*diff - 1)
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

