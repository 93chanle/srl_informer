import pandas as pd
import os
import sys
import numpy as np 

import warnings
warnings.filterwarnings('ignore')

import datetime as dt
from datetime import date, timedelta
import time

import urllib3
import requests
# Disable warnings in requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
s = requests.Session()

def download_data(market, data, date, return_path_to_file=False):
    
    url = f'https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/{data}?date={str(date)}&exportFormat=xlsx&market={market.upper()}&productTypes=aFRR'
       
    file = s.get(url, 
             verify=False)
    
    save_path =  f'raw/{market}_{data}/'
    
    if not os.path.exists(save_path): os.makedirs(save_path)

    save_name = f'{save_path}{market}_{data}_{date}.xlsx'
    
    with open(save_name, 'wb') as f:
        f.write(file.content)
    
    if return_path_to_file:
        return(save_name)
    
def preprocess_new_capacity_results(df:pd.DataFrame):
    """Preprocess new daily downloaded capacity_anonymousresults xlsx data, output data formats
    are optimized.

    Args:
        df (_type_): raw downloaded xlsx file
        
    """
    
    # Intial preprocessing

    df = (
        df
        .sort_values(by='DATE_TO', ascending=False)
        .rename(columns={'DATE_TO':'DATE',
                        df.columns[df.columns.str.startswith('CAPACITY')][0]:'CAPACITY_PRICE',
                        'OFFERED_CAPACITY_[MW]':'OFFERED_CAPACITY',
                        'ALLOCATED_CAPACITY_[MW]':'ALLOCATED_CAPACITY',
                        })
        .loc[df['COUNTRY'] == 'DE']
        .drop(columns=['DATE_FROM', 'NOTE','COUNTRY', 'TYPE_OF_RESERVES'])
        .reset_index(drop=True)
    )

    if df.DATE.unique()[0] < np.datetime64('2022-01-01'):
        df = (
            df
            .drop(columns=['ENERGY_PRICE_PAYMENT_DIRECTION', 'ENERGY_PRICE_[EUR/MWh]'])
        )

    # Create timestamp column (including date & product type)
    df.insert(loc=0, column='TIMESTAMP', value=df.apply(lambda row:row['DATE'] + timedelta(hours=int(row['PRODUCT'].split('_')[1])), axis=1))

    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    # Create columns
    df['MONTH'] = df['TIMESTAMP'].dt.month
    df['WEEKDAY'] = df['TIMESTAMP'].apply(lambda cell: cell.strftime('%a'))

    # Optimize data type
    df[['MONTH', 'WEEKDAY', 'PRODUCT']] = df[['MONTH', 'WEEKDAY', 'PRODUCT']].apply(lambda col: col.astype('category'), axis=0)
    df[['OFFERED_CAPACITY', 'ALLOCATED_CAPACITY']] = df[['OFFERED_CAPACITY', 'ALLOCATED_CAPACITY']].apply(lambda col: col.astype('int16'), axis=0)
    df['CAPACITY_PRICE'] = df['CAPACITY_PRICE'].astype('float32')

    # Lowercase column name
    df.rename(columns=dict(zip(
        df.columns, df.columns.str.lower()
    )), inplace=True)
    
    return(df)

df = pd.read_pickle('data/processed/SRL/capacity_anonymousresults_processed.pkl')

update_range=np.arange(df.date.max(), date.today()+timedelta(days=1), dtype="datetime64[D]")[1:]
print(f'Updating {len(update_range)} days: {update_range}')

if len(update_range) > 0:
    df_update = pd.DataFrame()
    for date in update_range:
        print(f'--- Crawling: {date} ---')
        file_path = download_data(market='capacity',data = 'anonymousresults', date=date, return_path_to_file=True)
        df_daily = pd.read_excel(file_path)
        df_daily = preprocess_new_capacity_results(df_daily)
        df_update = pd.concat([df_update, df_daily], axis=0)
        time.sleep(3) # Avoding sending requests
        
    df = pd.concat([df, df_update], axis=0)
df = (
    df
    .sort_values(by=['timestamp', 'capacity_price', 'allocated_capacity'])
)

df.to_pickle('data/processed/capacity_anonymousresults_processed.pkl')

# Capacity price data
df_cap = df.groupby(['date', 'product'], as_index=False).max('capacity_price')

# Divide products into separate CSV

for product in df_cap['product'].unique():
    df_ = df_cap[df_cap['product'] == product]
    df_.to_csv(f'data/processed/SRL_{product}.csv', index=False)
    
    