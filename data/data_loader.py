import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.utils.model_selection import train_test_split

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale='minmax', inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='capacity_price', scale='standard', inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        
        assert scale in ['minmax', 'standard', 'none']
        
        self.scale = scale        
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        if self.scale == 'minmax':
            self.scaler = MinMaxScaler()
          
        elif self.scale == 'standard':
            self.scaler = StandardScaler()
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        # This is the number of sequences we are getting from the input time series
        num_test = int(len(df_raw)*0.2) - self.pred_len
        num_vali = int(len(df_raw)*0.3)
        num_train = len(df_raw) - num_vali - num_test - self.seq_len - self.pred_len
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
            
            
        if self.scale == 'standard':
            train_data = df_data[0:num_train+self.seq_len]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        elif self.scale == 'none':
            data = df_data.values #np.array
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Create sequences based on the input, label and predict lengths
        self.seqs_x, self.seqs_y = self.generate_enc_dec_sequences(data)
        self.seqs_x_mark, self.seqs_y_mark = self.generate_enc_dec_sequences(data_stamp)   
        self.seqs_x_date, self.seqs_y_date = self.generate_enc_dec_sequences(df_stamp.date)
        
        # Set train-val-test borders (for sequences)
        # Here number of test and val are calculated first,
        # to make sure they stay the same regardless of input/label/pred len
        
        idx_seqs = [0, num_train, num_train + num_vali, len(self.seqs_x)]
        idx_data = [idx + self.seq_len for idx in idx_seqs] 
        idx_data[0] = 0
        
        # Subset sequences according to data type (train val test)
        self.seqs_x = self.seqs_x[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y  = self.seqs_y[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_x_mark = self.seqs_x_mark[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y_mark = self.seqs_y_mark[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_x_date = self.seqs_x_date[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y_date = self.seqs_y_date[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        
        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        # border2s = [num_train, num_train+num_vali, len(df_raw)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        
        # self.data_x = data[border1:border2]
        
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
        
        self.start_date_x = self.seqs_x_date[0].iloc[0]
        self.end_date_x = self.seqs_x_date[-1].iloc[-1]
        self.start_date_y = self.seqs_y_date[0].iloc[0]
        self.end_date_y = self.seqs_y_date[-1].iloc[-1]
        
        # if flag=='train':
        #     self.target_data_x = df_raw[[self.target]][]
        
        self.date_index_x = pd.date_range(self.start_date_x, self.end_date_x, freq='D')
        self.date_index_y = pd.date_range(self.start_date_y, self.end_date_y, freq='D')
        
        # Raw data for plotting
        self.target_data = df_raw[idx_data[self.set_type]: idx_data[self.set_type + 1]]
        self.target_date_range = pd.to_datetime(self.target_data['date'], format = '%m/%d/%Y')

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len 
    #     r_end = r_begin + self.label_len + self.pred_len
    
    #     seq_x = self.data_x[s_begin:s_end]
    #     if self.inverse:
    #         seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
    #     else:
    #         seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __getitem__(self, index):
        
        seq_x = self.seqs_x[index]
        seq_y  = self.seqs_y[index]
        seq_x_mark = self.seqs_x_mark[index]
        seq_y_mark = self.seqs_y_mark[index]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.seqs_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def generate_enc_dec_sequences(self, data):
        """_summary_

        Args:
            data (np.array): input data (row idx: time)

        Returns:
            list: 2 list of encoder & decoder inputs (with appropriate input, label & pred seq len)
        """
        enc_seqs=[]
        dec_seqs=[]
        
        inc_len=self.seq_len
        dec_len=self.label_len+self.pred_len
        
        for i in range(len(data) - self.seq_len - self.pred_len + 1): # Index purposes (otherwise error)
            enc_seq = data[i:i+inc_len]
            enc_seqs.append(enc_seq)
            
            dec_start_idx = i + self.seq_len- self.label_len
            # print(f'{dec_start_idx=}')
            dec_seq = data[dec_start_idx:dec_start_idx+dec_len]
            dec_seqs.append(dec_seq)
        
        return enc_seqs, dec_seqs
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SRL_XGBoost():
    def __init__(self, root_path, product_type):
        self.product_type = product_type
        self.root_path = root_path
        self.__read_data__()
        
    def __read_data__(self):
        data_path = f'{self.root_path}{self.product_type}.csv'
        
        df_raw = pd.read_csv(data_path)
        series = TimeSeries.from_dataframe(df_raw, 'date', 'capacity_price')
        
        self.train, temp = train_test_split(series, test_size=0.2)
        self.val, self.test = train_test_split(temp, test_size=0.5)
        