### Hyperparameter tuning, POSitive prices

conda activate thesis

python -u main_tuning.py --data SRL_POS_12_16 --data_path SRL_POS_12_16.csv --loss w_rmse

python -u main_tuning.py --data SRL_POS_16_20 --data_path SRL_POS_16_20.csv --loss w_rmse

python -u main_tuning.py --data SRL_POS_20_24 --data_path SRL_POS_20_24.csv --loss w_rmse

python -u main_tuning.py --data SRL_POS_00_04 --data_path SRL_POS_00_04.csv --loss rmse

python -u main_tuning.py --data SRL_POS_04_08 --data_path SRL_POS_04_08.csv --loss rmse

python -u main_tuning.py --data SRL_POS_08_12 --data_path SRL_POS_08_12.csv --loss rmse

python -u main_tuning.py --data SRL_POS_12_16 --data_path SRL_POS_12_16.csv --loss rmse

python -u main_tuning.py --data SRL_POS_16_20 --data_path SRL_POS_16_20.csv --loss rmse

python -u main_tuning.py --data SRL_POS_20_24 --data_path SRL_POS_20_24.csv --loss rmse