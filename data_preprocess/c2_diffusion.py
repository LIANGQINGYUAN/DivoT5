import json
import pandas as pd
from tqdm import tqdm

def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)
def save_data(df, o_name):
    df = df.astype(object)
    with open(f"{o_name}.json",'w+') as t:
        for i in tqdm(range(len(df))):
            item = df.iloc[i,:].to_dict()
            t.write(json.dumps(item)+'\n')
def save_dict(d, o_name):
    with open(f"{o_name}.json",'w+') as o:
        o.write(json.dumps(d))


data = read_data('./CommitPack_Diff_Java_V2.json')
def get_old(code):
    code_lines = code.split('\n')
    old = []
    for line in code_lines:
        if line.startswith('-') or line.startswith(' '):
            old.append(line[1:])
    return '\n'.join(old)

def get_new(code):
    code_lines = code.split('\n')
    new = []
    for line in code_lines:
        if line.startswith('+') or line.startswith(' '):
            new.append(line[1:])
    return '\n'.join(new)

def get_old_new(code):
    code_lines = code.split('\n')
    old = []
    new = []
    for line in code_lines:
        if line.startswith('-'):
            old.append(line[1:])
        if line.startswith('+'):
            new.append(line[1:])
        else:
            old.append(line[1:])
            new.append(line[1:])
    return '\n'.join(old), '\n'.join(new)

import random 
random.seed(0)

def get_diffusion(diff_list):
    temp_item_list = []
    new = get_new('\n'.join(diff_list[:])) # X0
    selected_num = 3
    if len(diff_list)<=selected_num:
        id_list = range(len(diff_list))
    else:
        id_list = [0] + sorted(random.sample(range(1, len(diff_list)), selected_num))
    # id_list = range(len(diff_list))
    for t in id_list: # from Xt to X1
        # old_first = get_old(diff_list[0])
        new_prefix = get_new('\n'.join(diff_list[:t]))
        old_suffix = get_old('\n'.join(diff_list[t:]))
        old = new_prefix + '\n' + old_suffix
        item['old'] = old
        item['new'] = new
        item['source_track'] = f'{i}_{t}'
        temp_item_list.append(item.copy())
        # print("new_prefix: ",new_prefix)
        # print("old_suffix: ",old_suffix)
        # print("old: ",old)
    return temp_item_list

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

item_list = []
for i, item in tqdm(data.iterrows(),total=len(data)):
    diff_info = item['diff']
    diff_list = diff_info.split('<@@DiffSegmentsSplit@@>')
    # print(len(diff_list))
    if len(diff_list) == 1:
        old, new = get_old_new(diff_list[0])
        item['old'] = old
        item['new'] = new
        item['source_track'] = '-1'
        item_list.append(item)
    else:
        if len(tokenizer.tokenize(''.join(diff_list)))<=510:
            item_list.extend(get_diffusion(diff_list))
        else: # >510
            # split diff_list
            diff_list_v2 = [d for d in diff_list if len(tokenizer.tokenize(d))<=510]
            if len(tokenizer.tokenize(''.join(diff_list_v2)))<510:
                item_list.extend(get_diffusion(diff_list_v2))
            else:
                length_diff = len(diff_list_v2)
                for j in range(length_diff//2):
                    diff_list_v3 = diff_list_v2[j*2:j*2+2]
                    item_list.extend(get_diffusion(diff_list_v3))
        # break

df = pd.DataFrame(item_list)

save_data(df[['commit','repos', 'msg', 'old','new']], 'CommitPack_Diff_Java_V2_Diffusion_T5_filtered_num_3')