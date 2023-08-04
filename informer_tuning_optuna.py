# %% [markdown]
# TUNING OPTUNA

# %%
from sklearn.metrics import mean_absolute_error
from utils.postprocessing import ProcessedResult
import optuna

from utils.metrics import LinEx, LinLin, weighted_RMSE, RMSE

SEED = 42

# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import argparse
import os
import torch
import yaml

import numpy as np
from datetime import datetime
from exp.exp_informer import Exp_Informer
from exp.args_parser import args_parsing

args = args_parsing()

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# now = ("19-07-2023_22-26-38")

Exp = Exp_Informer

# %%
args = args_parsing()

# %% [markdown]
# Define the objective function.
# 
# 

# %%
def objective(trial):
    
    torch.cuda.empty_cache()
    
    # param = {
    #     "objective": "binary",
    #     "metric": "auc",
    #     "verbosity": -1,
    #     "boosting_type": "gbdt",
    #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    #     "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    # }
    
    # SEARCH SPACE
    
    match args.loss:
        case 'linex':
            args.linex_weight = trial.suggest_float('linex_weight', 0.01, 3, step=0.01)
        case 'wrmse':
            args.wrmse_weight = trial.suggest_float('wrmse_weight', 1.0, 10.0, step=0.1)
        case 'linlin':
            args.linlin_weight = trial.suggest_float('linlin_weight', 0.05, 0.45, step=0.005)
    
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.train_epochs = trial.suggest_int("train_epochs", 6, 14)
    args.seq_len = trial.suggest_int('seq_len', 56, 112, step=7)
    label_seq_len_ratio = trial.suggest_float('label_seq_len_ratio', 0.4, 0.8, step=0.025)
    args.label_len = min(int(label_seq_len_ratio * args.seq_len), 77)
    args.e_layers = trial.suggest_int('e_layers', 2, 7)
    args.d_layers = trial.suggest_int('d_layers', 1, 4)
    args.n_heads = trial.suggest_int('n_heads', 4, 32, step=4)
    args.d_model = trial.suggest_int('d_model', 128, 1024, step=128)
    args.batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    
    # args.n_estimators = trial.suggest_int("n_estimators", 10, 100)
    # args.max_depth = trial.suggest_int("max_depth", 3, 12)
    # args.learning_rate = trial.suggest_float("learning_rate", 1e-1, 1e0, log=True)
    # args.min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    # args.gamma = trial.suggest_float('gamma', 0, 1, step=0.1)
    # args.subsample = trial.suggest_float('subsample', 0.5, 1.0, step=0.1)
    # args.colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1)
    # args.reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-5, 1.0)
    # args.reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-5, 1.0)

    exp = Exp(args)
    
    loss, revenue, _ = exp.tune()
    
    torch.cuda.empty_cache()
    
    # if trial.should_prune():s
    #     raise optuna.TrialPruned()
    
    # # Add a callback for pruning.
    # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    # gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

    # preds = gbm.predict(valid_x)
    # pred_labels = np.rint(preds)
    # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return loss, revenue

# %%

import logging
import sys

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = f"tune_informer_{args.data}_{args.loss}_{now}"  # Unique identifier of the study.
storage_name = "sqlite:///optuna_studies/{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,
                            load_if_exists=True,
                            directions=['minimize', 'maximize'],
                            sampler=optuna.samplers.TPESampler(seed=1993),
                            )

completes = 0

while completes < 100:
    
    study.optimize(objective, n_trials= (args.tune_num_samples - completes), catch=[Exception])

    st_df = study.trials_dataframe()
    if 'COMPLETE' in st_df['state'].values:
        completes = (st_df['state'] == 'COMPLETE').sum()

# study = optuna.create_study(
#     directions=['minimize', 'maximize'],
#     sampler=optuna.samplers.TPESampler(seed=SEED),
#     # pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
# )
# study.optimize(objective, n_trials=args.tune_num_samples, timeout=600)

# # %% [markdown]
# # ## Plot functions
# # Visualize the optimization history. See :func:`~optuna.visualization.plot_optimization_history` for the details.
# # 
# # 

# # %%
# study.get_trials()

# # %%
# optuna.visualization.plot_pareto_front(study, target_names=["loss", "revenue"])

# # %%
# plot_optimization_history(study, target=lambda t: t.values[1])

# # %% [markdown]
# # Visualize the learning curves of the trials. See :func:`~optuna.visualization.plot_intermediate_values` for the details.
# # 
# # 

# # %%
# plot_intermediate_values(study)

# # %% [markdown]
# # Visualize high-dimensional parameter relationships. See :func:`~optuna.visualization.plot_parallel_coordinate` for the details.
# # 
# # 

# # %%
# plot_parallel_coordinate(study,target=lambda t: t.values[1])

# # %% [markdown]
# # Select parameters to visualize.
# # 
# # 

# # %%
# plot_parallel_coordinate(study, params=["bagging_freq", "bagging_fraction"])

# # %% [markdown]
# # Visualize hyperparameter relationships. See :func:`~optuna.visualization.plot_contour` for the details.
# # 
# # 

# # %%
# plot_contour(study, target=lambda t: t.values[1])

# # %% [markdown]
# # Select parameters to visualize.
# # 
# # 

# # %%
# plot_contour(study, params=["bagging_freq", "bagging_fraction"])

# # %% [markdown]
# # Visualize individual hyperparameters as slice plot. See :func:`~optuna.visualization.plot_slice` for the details.
# # 
# # 

# # %%
# plot_slice(study)

# # %% [markdown]
# # Select parameters to visualize.
# # 
# # 

# # %%
# plot_slice(study, params=["bagging_freq", "bagging_fraction"])

# # %% [markdown]
# # Visualize parameter importances. See :func:`~optuna.visualization.plot_param_importances` for the details.
# # 
# # 

# # %%
# plot_param_importances(study)

# # %% [markdown]
# # Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
# # 
# # 

# # %%
# optuna.visualization.plot_param_importances(
#     study, target=lambda t: t.duration.total_seconds(), target_name="duration"
# )

# # %% [markdown]
# # Visualize empirical distribution function. See :func:`~optuna.visualization.plot_edf` for the details.
# # 
# # 

# # %%
# plot_edf(study)


