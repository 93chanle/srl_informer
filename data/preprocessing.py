import sys
import os

root = os.path.join(sys.path[0].removesuffix('\\data'))
sys.path.append(root)

from data.external_data_conversion import convert_external_data_finanzen, add_external_data_to_srl

convert_external_data_finanzen('gas')
convert_external_data_finanzen('coal')

products=[
    'SRL_NEG_00_04',
    'SRL_NEG_00_04',
    'SRL_NEG_04_08',
    'SRL_NEG_08_12',
    'SRL_NEG_12_16',
    'SRL_NEG_16_20',
    'SRL_NEG_20_24',
    'SRL_POS_00_04',
    'SRL_POS_04_08',
    'SRL_POS_08_12',
    'SRL_POS_12_16',
    'SRL_POS_16_20',
    'SRL_POS_20_24',
]

for product in products:
    add_external_data_to_srl(srl_product_name=product,
                             external_data_names=['gas','coal'])    
    print('External data added!')