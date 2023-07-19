import argparse
import os
import torch
import yaml
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.search.optuna import OptunaSearch

import numpy as np
from datetime import datetime
from exp.exp_informer import Exp_Informer
from exp.args_parser import args_parsing

args = args_parsing()

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

Exp = Exp_Informer

### TUNING 

# Define search space
# Create Tuner object
# Define objective function

# Define search space 
search_space = {'learning_rate': tune.loguniform(1e-5, 1e-2), 
                'train_epochs': tune.randint(6, 14),
                'seq_len': tune.choice([56, 70, 84, 98, 112]),
                'label_seq_len_ratio': tune.quniform(0.4, 0.8, 0.025),
                'e_layers': tune.randint(2, 7),
                'd_layers': tune.randint(1, 4),
                'n_heads': tune.choice([4, 8, 12, 16, 24, 32]),
                'd_model': tune.choice([128, 256, 512, 1024]),
                'batch_size': tune.choice([4, 8, 16, 32]),
                }

match args.loss:
    case 'linex':
        search_space['linex_weight'] = tune.quniform(0.01, 3, 0.005)
    case 'w_rmse':
        search_space['w_rmse_weight'] = tune.quniform(1, 10, 0.1)
    case 'linlin':
        search_space['linlin_weight'] = tune.quniform(0.05, 0.45, 0.025)  

# Define trainable
# exp.tune() serves as objective function for Ray
def trainable(config):

    args.learning_rate = config["learning_rate"]
    args.train_epochs = config["train_epochs"]
    args.seq_len = config['seq_len']
    args.label_len = min(int(config['label_seq_len_ratio'] * args.seq_len), 77)
    args.e_layers = config['e_layers']
    args.d_layers = config['d_layers']
    args.n_heads = config['n_heads']
    args.d_model = config['d_model']
    args.batch_size = config['batch_size']
    
    match args.loss:
        case 'linex':
            args.linex_weight = config['linex_weight']
        case 'w_rmse':
            args.w_rmse_weight = config['w_rmse_weight']
        case 'linlin':
            args.linlin_weight = config['linlin_weight']

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
        num_samples=args.tune_num_samples,
        trial_dirname_creator=my_trial_dirname_creator,
    ),
    run_config=air.RunConfig(
        name=f'tune_{args.data}_{args.loss}_{now}',
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
