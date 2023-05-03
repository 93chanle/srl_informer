import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import weighted_RMSE
import matplotlib.dates as mdates
import torch

from utils.metrics import RMSE, MSE

class ProcessedResult():
    def __init__(self, preds, trues, train_scaler, args, data):
        self.scaler = train_scaler
        self.args = args
        self.pred_len = preds.shape[0]
        self.pred_raw = self.convert_seq(preds, inverse=False)
        self.true_raw = self.convert_seq(trues, inverse=False)
        self.pred = self.convert_seq(preds, inverse=True)
        self.true = self.convert_seq(trues, inverse=True)
        self.pred_naive = self.true.shift(1)
        self.data = data
    
    def convert_seq(self, seq_raw, inverse=True):
        if inverse: 
            seq = self.scaler.inverse_transform(seq_raw)
        else: seq = seq_raw
        
        if seq.shape[1] == 1:
            return pd.Series(seq.mean(1).squeeze())
            
        else:
            array = seq.squeeze()
            array = np.array([np.concatenate([np.repeat(np.nan, i), array[i], np.repeat(np.nan, self.pred_len-i-1)]) for i in np.arange(self.pred_len)])
            df = pd.DataFrame(array.transpose())
            return df.mean(axis=1)

    def plot_pred_vs_true(self, pred):
        fig, ax = plt.subplots(figsize=(12,6))
        pred_non_neg = np.where(pred < 0, 0, pred)
        
        ax.plot(self.data.date_index, self.true, label='True')
        ax.plot(self.data.date_index, pred, label ='Raw prediction', linestyle ='--', alpha = 0.3)
        ax.plot(self.data.date_index, pred_non_neg, label ='Predicted SRL price')
        
        # plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€', 
        #              xy=(0.05, 0.9), xycoords='axes fraction',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5))
        # plt.annotate(f'Weighted RMSE (alpha={self.args.alpha}): {self.weighted_rmse(pred)}', 
        #              xy=(0.05, 0.8), xycoords='axes fraction',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5))
        
        plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€, Baseline revenue: {self.predict_revenue(self.pred_naive)}€, Weighted RMSE (alpha={self.args.alpha}): {self.weighted_rmse(pred)}', 
                     xy=(0.1, -0.1), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.2))
        
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Title
        ax.set_title(f'{self.args.data}, on validation set, from {self.data.start_date} to {self.data.end_date}')

        plt.close()
        return(fig)
    
    def predict_revenue(self, pred):
        pred_non_neg = np.where(pred < 0, 0, pred)
        return np.nansum(np.where(pred_non_neg > self.true, 0, pred_non_neg)).round(2)
    
    def weighted_rmse(self, pred):
        result = weighted_RMSE(pred, self.true, self.args.alpha)
        return result
    
    
class ProcessedResultXGB():
    def __init__(self, pred, true, args, dataset):
        self.args = args
        self.pred = pred
        self.true = true
        self.pred_naive = self.true.shift(1, fill_value=true[0])
        self.dataset = dataset
    
    def plot_pred_vs_true(self, pred):
        fig, ax = plt.subplots(figsize=(12,6))
        pred_non_neg = np.where(pred < 0, 0, pred)
        
        date_index = self.dataset.val.time_index
        
        ax.plot(date_index, self.true, label='True')
        ax.plot(date_index, pred, label ='Raw prediction', linestyle ='--', alpha = 0.3)
        ax.plot(date_index, pred_non_neg, label ='Predicted SRL price')
        
        plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€, Baseline revenue: {self.predict_revenue(self.pred_naive)}€', 
                     xy=(0.1, -0.1), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.2))
        
        #, Weighted RMSE (alpha={self.args.alpha}): {self.weighted_rmse(pred)}
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Title
        ax.set_title(f'{self.dataset.product_type}, on validation set, from {date_index[0]} to {date_index[-1]}')

        plt.close()
        return(fig)
    
    def predict_revenue(self, pred):
        pred_non_neg = np.where(pred < 0, 0, pred)
        return np.nansum(np.where(pred_non_neg > self.true, 0, pred_non_neg)).round(2)
    
    def weighted_rmse(self, pred):
        result = weighted_RMSE(pred, self.true, self.args.alpha)
        return result
    
    def rmse(self, pred):
        result = RMSE(pred, self.true)
        return result