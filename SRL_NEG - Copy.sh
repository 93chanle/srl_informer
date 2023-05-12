### Hyperparameter tuning, negative prices

conda activate thesis

python -u main_tuning.py --data SRL_NEG_00_04 --data_path SRL_NEG_00_04.csv --loss linlin

python -u main_tuning.py --data SRL_NEG_04_08 --data_path SRL_NEG_04_08.csv --loss linlin

python -u main_tuning.py --data SRL_NEG_08_12 --data_path SRL_NEG_08_12.csv --loss linlin

python -u main_tuning.py --data SRL_NEG_12_16 --data_path SRL_NEG_12_16.csv --loss linlin

python -u main_tuning.py --data SRL_NEG_16_20 --data_path SRL_NEG_16_20.csv --loss linlin

python -u main_tuning.py --data SRL_NEG_20_24 --data_path SRL_NEG_20_24.csv --loss linlin