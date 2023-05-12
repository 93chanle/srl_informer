
import pandas as pd
import numpy as np
import os

def convert_external_data_finanzen(data_name, fillna=True, save_csv=True):

    """Converts raw data from finanzen.de
    Applicable for gas & coal prices

    Args:
        filedir (str): directory to file + file name
    """
    
    path = os.path.join('data', 'raw', 'external_data', f'{data_name}.xlsx')

    df = pd.read_excel(path, 
                        dtype={'Datum': 'object', 
                                # 'Schluss': 'float',
                                # 'Eröffnung': 'float',
                                # 'Tageshoch': 'float',
                                # 'Tagestief': 'float',
                                })

    df['Schluss'] = df['Schluss'].str.replace(',','.').astype('float64')
    df = df.sort_values(by='Datum').reset_index(drop=True)
    
    df = df[['Datum', 'Schluss']]
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y').sort_values().reset_index(drop=True)

    range = pd.date_range(df['Datum'].iloc[0], df['Datum'].iloc[-1])
    for date in range:
        if date not in list(df['Datum']):
            df_ = pd.DataFrame({
                'Datum': [date],
                'Schluss': [np.NaN],
                # 'Eröffnung': [np.NaN],
                # 'Tageshoch': [np.NaN],
                # 'Tagestief': [np.NaN]
            })
            df = df.append(df_, ignore_index=True)
            
    df = df.sort_values(by='Datum').reset_index(drop=True)
    df.columns = df.columns.str.lower()
    
    if fillna:
        df = df.fillna(method='ffill')
        
    df.columns = ['date', f'{data_name}']
    
    if save_csv:
        df.to_csv(f'data/processed/external_data/{data_name}.csv', index=False)
        print('Processed data saved!')
    
    else: return df
    
    
def add_external_data_to_srl(srl_product_name = 'SRL_NEG_00_04', 
                            external_data_names = ['gas'],
                            save_csv=True):
    
    assert type(external_data_names) in [str, list]
    
    if type(external_data_names) == str:
        external_data_names = [external_data_names]
        
    root_path= os.path.join('data', 'processed')

    df = pd.read_csv(os.path.join(root_path, 'SRL', f'{srl_product_name}.csv'))
    
    for name in external_data_names:
        ex = pd.read_csv(os.path.join(root_path, 'external_data', f'{name}.csv'))
        df = df.merge(ex, on='date')
        
    try:
        assert not df.isna().any().all()
    
    except AssertionError as e:
        print('NA values in merged dataset')
        
    if save_csv:
        df.to_csv(f'data/processed/SRL/{srl_product_name}.csv', index=False)
        
    return df