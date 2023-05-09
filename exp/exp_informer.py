from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from utils.postprocessing import ProcessedResult

from utils.tools import EarlyStopping, adjust_learning_rate, EarlyStoppingNoSaveModel
from utils.metrics import metric, WeightedRMSE, weighted_RMSE, LinExLoss

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import matplotlib.pyplot as plt

import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        # data_dict = {
        #     'ETTh1':Dataset_ETT_hour,
        #     'ETTh2':Dataset_ETT_hour,
        #     'ETTm1':Dataset_ETT_minute,
        #     'ETTm2':Dataset_ETT_minute,
        #     'WTH':Dataset_Custom,
        #     'ECL':Dataset_Custom,
        #     'Solar':Dataset_Custom,
        #     'custom':Dataset_Custom,
        # }
        # Data = data_dict[self.args.data]
        
        Data = Dataset_Custom
        
        timeenc = 0 if args.embed!='timeF' else 1

        assert flag in ['train', 'val', 'test']
        if flag in ['test', 'val']:
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        # elif flag=='pred':
        #     shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        #     Data = Dataset_Pred
        elif flag == 'train':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scale=args.scale,
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        
        # PRINTSTAT
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        
        # criterion =  LinExLoss(self.args.linex_weight)
        match self.args.loss:
            case 'linex':
                criterion=LinExLoss(self.args.linex_weight)
            case 'w_rmse':
                criterion=WeightedRMSE(self.args.w_rmse_weight)
            case 'rmse':
                criterion=nn.MSELoss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            try:
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
            except RuntimeError as e:
                print('Wrong!')
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def report_tune(self, vali_data, vali_loader, criterion):
        train_data, _ = self._get_data(flag = 'train')
        
        self.model.eval()
        total_loss = []
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            total_loss.append(loss)
            
        total_loss = np.average(total_loss)
        
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        
        # Create result object
        result = ProcessedResult(preds=preds, trues=trues, 
                                 train_scaler=train_data.scaler,
                                 args=self.args,
                                 data=vali_data)
        predicted_revenue = result.predict_revenue(result.pred)
        fig = result.plot_pred_vs_true(result.pred)
        fig.savefig(f'informer.png', bbox_inches = 'tight')
        fig = result.plot_pred_vs_true(result.pred_naive)
        fig.savefig(f'naive.png', bbox_inches = 'tight')
        
        # Dump result object
        with open('processed_result.pickle', 'wb') as f:
            pkl.dump(result, f)
        
        # self.model.train()
        return total_loss, predicted_revenue

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        # Make path for checkpoint
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, data_type='vali'):
        train_data, train_loader = self._get_data(flag = 'train')
        if data_type == 'vali':
            test_data, test_loader = self._get_data(flag = 'val')
        elif data_type == 'test':
            test_data, test_loader = self._get_data(flag = 'test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # Shape: (batch) size x pred_len x output_feature (price)
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        
        # print('test shape:', preds.shape, trues.shape)

        # Make path for results
        folder_path = './results/' + self.args.timestamp + "_" + self.args.data +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # Save prediction
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        
        # Create result object
        result = ProcessedResult(preds=preds, trues=trues, 
                                 args=self.args,
                                 data=test_data)   
        # Dump result object
        with open('processed_result_test.pkl', 'wb') as f:
            pkl.dump(result, f)
        
        predicted_revenue = result.predict_revenue(result.pred)
        fig = result.plot_pred_vs_true(result.pred)
        fig.savefig(folder_path + 'informer_result.png', bbox_inches='tight')
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
    
    
    def tune(self):
            
        # Retrieve different data & loader here in the train loop
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping_no_save = EarlyStoppingNoSaveModel(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                
                # Investigate output
                # Out has size of [batchsize, pred_len, target]
                # print(f'Pred is type: {type(pred)} and has shape {pred.shape}')
                # print(f'True is type: {type(true)} and has shape {true.shape}')
                # print(f'Loss is {loss}, type: {type(loss)} and has shape {loss.shape}')
                
                train_loss.append(loss.item())
                
                # if (i+1) % 100==0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time()-time_now)/iter_count
                #     left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_revenue = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Vali Revenue: {5:.7f} Test Revenue: {5:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss, vali_revenue, test_revenue))
            early_stopping_no_save(vali_loss, self.model)
            if early_stopping_no_save.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        
        #Report final loss & revenue
        tune_loss, tune_revenue = self.report_tune(vali_data, vali_loader, criterion)
            
        # Turn off saving model
        
        # best_model_path = path+'/'+'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        
        return tune_loss, tune_revenue, self.model
