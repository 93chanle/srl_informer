import argparse
import os
import torch
import yaml
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.search.optuna import OptunaSearch

import numpy as np
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')


# non-tuneable
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, required=False, default='SRL_NEG_00_04', help='data')
parser.add_argument('--root_path', type=str, default="C:/codes/srl_informer/data/processed/SRL/", help='root path of the data file')
parser.add_argument('--data_path', type=str, default='SRL_NEG_00_04.csv', help='data file')    
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='capacity_price', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--save_checkpoints', type=str, default=False, help='whether to save model checkpoint')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--activation', type=str, default='gelu',help='activation')

# tuneable
parser.add_argument('--seq_len', type=int, default=4, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=3, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type') # Either padded by 0 or 1
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--alpha', type=float, default=10,help='weighted parameter for loss function')
parser.add_argument('--linex_weight', type=float, default=0.001,help='weighted parameter for linear-exponential loss function')


args = parser.parse_args()

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

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

### TUNING 

# Define search space
# Create Tuner object
# Define objective function

# Define search space 
search_space = {'learning_rate': tune.loguniform(1e-5, 1e-2), 
                'train_epochs': tune.randint(6, 14),
                'alpha': tune.quniform(1, 10, 0.1),
                'seq_len': tune.choice([56, 70, 84, 98, 112]),
                'label_seq_len_ratio': tune.quniform(0.4, 0.8, 0.025),
                'e_layers': tune.randint(2, 7),
                'd_layers': tune.randint(1, 4),
                'n_heads': tune.choice([4, 8, 12, 16, 24, 32]),
                'd_model': tune.choice([128, 256, 512, 1024]),
                'batch_size': tune.choice([4, 8, 16, 32]),
                'linex_weight': tune.quniform(0.001, 3, 0.001),
                }

# Define trainable
# exp.tune() serves as objective function for Ray
def trainable(config):
        
    args.learning_rate = config["learning_rate"]
    args.train_epochs = config["train_epochs"]
    args.alpha = config['alpha']
    args.seq_len = config['seq_len']
    args.label_len = min(int(config['label_seq_len_ratio'] * args.seq_len), 77)
    args.e_layers = config['e_layers']
    args.d_layers = config['d_layers']
    args.n_heads = config['n_heads']
    args.d_model = config['d_model']
    args.batch_size = config['batch_size']
    args.linex_weight = config['linex_weight']

    print(f'--------------Start new run-------------------')
    # print(f'Tune learning rate: {args.learning_rate}')
    # print(f'Tune train epochs: {args.train_epochs}')
    # print(f'Tune alpha: {args.alpha}')
    # print(f'Tune seq_len: {args.seq_len}')
    # print(f'Tune pred_len: {args.pred_len}')

    # setting = '{}_{}_alpha{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(args.model, args.data, args.alpha, args.features, 
    #             args.seq_len, args.label_len, args.pred_len,
    #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
    #             args.embed, args.distil, args.mix, args.des)
    # setting = setting + '_' + now
    
    exp = Exp(args)
    
    tune_loss, tune_revenue, model = exp.tune() # Compute metric
    session.report({"revenue": tune_revenue, 
                    "loss": tune_loss,
                    },
                   checkpoint=Checkpoint.from_dict(
                       dict(model=model.state_dict()))
                   )  # Report to Tune
    
    print(f'Predicted revenue: {tune_revenue}') 
    print(f'Loss: {tune_loss}') 
    print(f'--------------End run-------------------')
    
# Define search algorithm
algo = OptunaSearch(metric=["loss", "revenue"], mode=["min", "max"])

# Custom trial name creator
def my_trial_dirname_creator(trial):
    trial_name = trial.trainable_name
    trial_id = trial.trial_id
    return f"my_prefix_{trial_name}_{trial_id}"

# Start Tune run
tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=200,
        trial_dirname_creator=my_trial_dirname_creator,
    ),
    run_config=air.RunConfig(
        name=f'tune_{args.data}_{now}',
        local_dir='ray_tune/',
        stop={"training_iteration": 100},
        checkpoint_config=air.CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute='loss',
            checkpoint_score_order='min',
        )
    ),
    param_space=search_space,
)

results = tuner.fit()

best_loss = results.get_best_result(metric='loss', mode='min')

best_loss.config

best_revenue = results.get_best_result(metric='revenue', mode='max')

best_revenue.config

# Delete all other trials
folder_path = f'C:/codes/srl_informer/ray_tune/tune_{args.data}_{now}'

trainable_to_keep = [best_loss.log_dir.parts[-1], best_revenue.log_dir.parts[-1]]

# Get a list of all subfolders in the folder
subfolders = os.listdir(folder_path)

# Iterate over the subfolders and delete each one except for the folder to keep
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path) and subfolder not in trainable_to_keep and subfolder.find('.') == -1:
        os.system('rm -rf {}'.format(subfolder_path))
