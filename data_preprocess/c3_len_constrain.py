import json
import pandas as pd
from tqdm import tqdm
import sys

def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)
def save_data(df, o_name, suffix = 'json'):
    df = df.astype(object)
    with open(f"{o_name}.{suffix}",'w+') as t:
        for i in tqdm(range(len(df))):
            item = df.iloc[i,:].to_dict()
            t.write(json.dumps(item)+'\n')
def save_dict(d, o_name):
    with open(f"{o_name}.json",'w+') as o:
        o.write(json.dumps(d))

filename = 'CommitPack_Diff_Java_V2_Diffusion_T5_filtered_num_3_with_mask'
data_cp = read_data(f'./{filename}.json')

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

selected = []
for i, item in tqdm(data_cp.iterrows(), total=len(data_cp)):
    # if len(item['msg'].split())>=3 and len(item['old'])<=1000 and len(item['new'])<=1000:
    if len(item['msg'].split())>=3 and len(tokenizer.tokenize(item['old'])) <= 510 and len(tokenizer.tokenize(item['new'])) <= 510:
        selected.append(1)
    else:
        selected.append(0)

data_cp['selected'] = selected
data_cp = data_cp.loc[data_cp.selected==1]
data_cp.reset_index(drop=True, inplace=True)
save_data(data_cp, f'{filename}_length_constrain') 