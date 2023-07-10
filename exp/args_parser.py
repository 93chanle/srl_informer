import argparse
import os
import yaml
import torch

import numpy as np
from datetime import datetime

debug = False

def args_parsing():

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    # Main arguements
    parser.add_argument('--data', type=str, required=False, default='SRL_NEG_00_04', help='data')
    parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--loss', type=str, default='linlin',help='customized loss functions, one of [w_rmse, linex, linlin, rmse]')

    parser.add_argument('--w_rmse_weight', type=float, default=5,help='weighted parameter for weighted rmse loss function')
    parser.add_argument('--linex_weight', type=float, default=0.05,help='weighted parameter for linear-exponential loss function')
    parser.add_argument('--linlin_weight', type=float, default=0.1,help='weighted parameter for linlin / pinball loss function')

    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=3, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    parser.add_argument('--timestamp', type=str, default=now)
    
    # C:\\codes\\srl_informer\\
    parser.add_argument('--root_path', type=str, default= 'data\\processed\\SRL\\', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SRL_NEG_00_04.csv', help='data file')    
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--cols', type=str, nargs='+', help='external col names from the data files as the additional input features (not including target)')
    
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    
    parser.add_argument('--scale', type=str, default='standard', help='forecasting task, options: [standard, minmax, none]')
    parser.add_argument('--target', type=str, default='capacity_price', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    # Input size (starts encoding) (also relates to number of input features)
    # Case of univariate := 1, only SRL price
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    
    # Output size (for what?)
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    
    # Dim of model
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    
    # No of attention head, layers
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    
    # Only relevant for stacked Informer
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    
    # Testing the distil feature?
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    
    
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')

    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
    
    # FOR TUNING
    parser.add_argument('--tune_num_samples', type=int, default=200, help='Number of sample interations in hyperparameter tuning')

    args = parser.parse_args("")
    
    if debug:
        args.data = 'SRL_NEG_00_04_dummy'
        args.loss = 'rmse'
        args.seq_len = 4
        args.label_len = 3
        args.pred_len = 1
        args.d_model = 16
        args.d_ff = 10
        args.data_path = 'SRL_NEG_00_04_dummy.csv'
        args.itr = 1
        args.train_epochs = 3
        args.scale = 'none'
        args.factor = 1
        
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'SRL_NEG_00_04':{'data':'SRL_NEG_00_04.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_NEG_04_08':{'data':'SRL_NEG_04_08.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_NEG_08_12':{'data':'SRL_NEG_08_12.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_NEG_12_16':{'data':'SRL_NEG_12_16.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_NEG_16_20':{'data':'SRL_NEG_16_20.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_NEG_20_24':{'data':'SRL_NEG_20_24.csv','T':'capacity_price','S':[1,1,1]},
        
        'SRL_POS_00_04':{'data':'SRL_POS_00_04.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_POS_04_08':{'data':'SRL_POS_04_08.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_POS_08_12':{'data':'SRL_POS_08_12.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_POS_12_16':{'data':'SRL_POS_12_16.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_POS_16_20':{'data':'SRL_POS_16_20.csv','T':'capacity_price','S':[1,1,1]},
        'SRL_POS_20_24':{'data':'SRL_POS_20_24.csv','T':'capacity_price','S':[1,1,1]},
    }

    # if args.data in data_parser.keys():
    #     data_info = data_parser[args.data]
    #     args.data_path = data_info['data']
    #     args.target = data_info['T']
    #     args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]


    # Pass default values for external data incorporation
    if args.features == 'MS' and args.cols is None:
        args.cols = ['gas', 'coal']

    args.root_path = os.path.abspath(args.root_path)

    print('Args in experiment:')
    print(args)
    print('')
    return args

# with open("args.yaml", "w") as f:
#     yaml.dump(args, f)

