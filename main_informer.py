import argparse
import os
import torch
import yaml

import numpy as np
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

from exp.exp_informer import Exp_Informer

from exp.args_parser import args_parsing

# For saving model structure
import torch.onnx

# Parsing args
args = args_parsing()

# Create experiment
Exp = Exp_Informer
print('Created Informer experiment')

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    
    # Export for debugging
    # import pickle as pkl
    # with open('data\\dummy_dataset\\dummy_model.pkl', 'wb') as f:
    #     pkl.dump(exp, f)
    # with open('data\\dummy_dataset\\dummy_model_args.pkl', 'wb') as f:
    #     pkl.dump(args, f)
        
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # criterion =  WeightedRMSE(args.alpha)
    # vali_data, vali_loader = exp._get_data(flag = 'val')
    
    # exp.report_tune(vali_data, vali_loader, criterion)
    
    exp.test(setting, data_type='vali')

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

