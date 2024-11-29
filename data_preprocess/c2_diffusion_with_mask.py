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

import random
random.seed(42)

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('/Salesforce/codet5-base')

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

import json
import pandas as pd
from tqdm import tqdm
import sys
import re 
from mask.mask import span_mask
import numpy as np
import random
from mask import diff_utils

def insert_token(tokens, index, new_token):
    tokens.insert(index, new_token)

def delete_token(tokens, index):
    del tokens[index]

def replace_token(tokens, index, new_token):
    tokens[index] = new_token

def get_nl_mask(nl_tokens):
    nl_length = len(nl_tokens)
    if nl_length>5:
        mean_span = 2.5
        mean_sapn_length = max(int(nl_length*0.2//mean_span), 1)
        masktags = span_mask.random_spans_noise_mask(nl_length, mean_span, mean_sapn_length)
        tokens_with_mask = [t if m==0 else '<mask>' for m, t in zip(masktags, nl_tokens)]
        return tokens_with_mask
    return nl_tokens


def get_span_mask_from_mask(mask, tokens, sid=0):
    SPECIAL_ID = sid
    spmask = []
    for i in range(len(mask)):
        if i==0:
            if mask[i] == '<mask>' and SPECIAL_ID<99:
                spmask.append(f"<extra_id_{SPECIAL_ID}>")
                SPECIAL_ID+=1
            else:
                spmask.append(mask[i])
        else:
            if mask[i] == '<mask>' and mask[i-1] != '<mask>' and SPECIAL_ID<99:
                spmask.append(f"<extra_id_{SPECIAL_ID}>")
                SPECIAL_ID+=1
            elif mask[i] != '<mask>':
                spmask.append(mask[i])
        # print(SPECIAL_ID)
        # print("spmask: ", mask[i], tokens[i], spmask[-1], "**All: **",spmask)
    return spmask, SPECIAL_ID

def get_keep_span_mask(o_tokens, o_labels, n_tokens, ctype='<java>', nl_tokens=None, SPECIAL_ID=0):
    length_keep = len(o_labels) - sum(o_labels)
    # print(length_keep)
    mean_span = 2.5
    mean_sapn_length = max(int(length_keep*0.3//mean_span), 1)
    masktags = span_mask.random_spans_noise_mask(length_keep, mean_span, mean_sapn_length)
    # print(masktags)
    masktags_keep = []
    index = 0
    for l in o_labels:
        if l !=0:
            masktags_keep.append(0)
        else:
            masktags_keep.append(masktags[index])
            index+=1
    # print(masktags_keep)
    tokens_with_mask = [t if m==0 else '<mask>' for m, t in zip(masktags_keep, o_tokens)]
    tokens_with_mask, SPECIAL_ID = get_span_mask_from_mask(tokens_with_mask, o_tokens, SPECIAL_ID)
    nl_tokens_mask = nl_tokens
    input_mask_tokens = ['<msg>'] + nl_tokens_mask + [ctype] + tokens_with_mask
    input_original_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    # print("input_original_tokens: ",input_original_tokens)
    # print("input_mask_tokens:     ",input_mask_tokens)
    input_tokens, SPECIAL_ID = get_span_mask_from_mask(input_mask_tokens, input_original_tokens, SPECIAL_ID)

    output_labels = n_tokens
    return input_tokens, output_labels, SPECIAL_ID

def corrupt_code_v2(o_tokens, insert_prob, delete_prob, replace_prob, SPECIAL_ID = 0):
    SPECIAL_ID = SPECIAL_ID
    tokens = o_tokens
    # SPECIAL_ID = 0
    for index, token in enumerate(tokens):
        rand = random.uniform(0, 1)
        if rand < insert_prob[1] and rand > insert_prob[0] and SPECIAL_ID<99:
            new_token = f"<extra_id_{SPECIAL_ID}>"
            insert_token(tokens, index, new_token)
            SPECIAL_ID+=1
            continue
        if rand < delete_prob[1] and rand > delete_prob[0]:
            delete_token(tokens, index)
            continue
        if rand < insert_prob[1] and rand > insert_prob[0] and SPECIAL_ID<99:
            new_token = f"<extra_id_{SPECIAL_ID}>"
            replace_token(tokens, index, new_token)
            SPECIAL_ID+=1
            continue
    return tokens, SPECIAL_ID

def get_corrupted_pred_v2(o_tokens, n_tokens, ctype='<java>', nl_tokens=None, SPECIAL_ID=0): # subtokens
    code_delete_prob = (0, 0.495/3)
    code_insert_prob = (0.495/3, 0.71/3)
    code_replace_prob = (0.71/3, 1/3)
    try:
        corrupted_tokens, SPECIAL_ID = corrupt_code_v2(o_tokens, code_insert_prob, code_delete_prob,  code_replace_prob, SPECIAL_ID)
    except:
        return None
    if nl_tokens:
        input_tokens = ['<msg>'] + nl_tokens + [ctype] + corrupted_tokens
    else:
        input_tokens =  corrupted_tokens
    output_labels = n_tokens
    return input_tokens, output_labels, SPECIAL_ID

def get_target_generation(o_tokens, n_tokens, ctype='<java>', nl_tokens=None, SPECIAL_ID=0):
    input_tokens = []
    # input_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens

    nl_tokens_mask = nl_tokens
    o_tokens_mask = get_nl_mask(o_tokens)
    input_mask_tokens = ['<msg>'] + nl_tokens_mask + [ctype] + o_tokens_mask
    input_original_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    # print("input_mask_tokens:", input_mask_tokens)
    # print("input_original_tokens:", input_original_tokens)
    input_tokens, SPECIAL_ID = get_span_mask_from_mask(input_mask_tokens, input_original_tokens, SPECIAL_ID)

    output_labels = n_tokens
    return input_tokens, output_labels, SPECIAL_ID


def get_old_new_tokens_labels(old, new, tokenizer):
    # ex_token_diff = diff_utils.compute_code_diffs(ctu.get_tokenstr_list(ctu.tokenize_code(old)), ctu.get_tokenstr_list(ctu.tokenize_code(new)))[1]
    ex_token_diff = diff_utils.compute_code_diffs(tokenizer.tokenize(old), tokenizer.tokenize(new))[1]
    labels = [ex_token_diff[i] for i in range(len(ex_token_diff)) if i%2 ==0]
    tokens = [ex_token_diff[i] for i in range(len(ex_token_diff)) if i%2 ==1]
    assert len(labels) == len(tokens), "The length of lables and tokens are not equal!"
    o_tokens = []
    o_labels = []
    n_tokens = []
    n_labels = []
    for t, l in zip(tokens, labels):
        if l=="<KEEP>":
            o_tokens.append(t)
            o_labels.append(0)
            n_tokens.append(t)
            n_labels.append(0)
        elif "NEW" in l or "INSERT" in l:
            n_tokens.append(t)
            n_labels.append(2)
        elif "OLD" in l or "DELETE" in l:
            o_tokens.append(t)
            o_labels.append(1)
        else:
            print("Split Fail!")
    return o_tokens, o_labels, n_tokens, n_labels


def get_diffusion(old_new_list, item):
    temp_item_list = []
    new = '\n'.join([i[1] for i in old_new_list]) # X0
    selected_num = 3
    if len(old_new_list)<=selected_num:
        id_list = range(len(old_new_list))
    else:
        id_list = [0] + sorted(random.sample(range(1, len(old_new_list)), selected_num))
    # id_list = range(len(diff_list))
    for t in id_list: # from Xt to X1
        # old_first = get_old(diff_list[0])
        # new_prefix = get_new('\n'.join(diff_list[:t]))
        new_prefix = '\n'.join([i[1] for i in old_new_list[:t]])
        # old_suffix = get_old('\n'.join(diff_list[t:]))
        old_suffix = '\n'.join([i[0] for i in old_new_list[t:]])
        old = new_prefix + '\n' + old_suffix
        item['old'] = old
        item['new'] = new
        item['source_track'] = f'{i}_{t}'
        temp_item_list.append(item.copy())
        # print("new_prefix: ",new_prefix)
        # print("old_suffix: ",old_suffix)
        # print("old: ",old)
    return temp_item_list

import signal
def handler(signum, frame):
    raise Exception("Time's up!")
signal.signal(signal.SIGALRM, handler)


with open('CommitPack_Diff_Java_V2_Diffusion_T5_filtered_num_3_with_mask.json', 'a+') as file:
    for i in tqdm(range(len(data))):
        item_list = []
        ex_diff = data['diff'][i]
        item = data.iloc[i,:].to_dict()
        diff_list = ex_diff.split('<@@DiffSegmentsSplit@@>')
        sp_id=0
        # get masked old code
        old_new_list = []
        try:
            signal.alarm(10)
            for di in range(len(diff_list)):
                try:
                    r_num = random.random()
                    old, new = get_old_new(diff_list[di])
                    o_tokens, o_labels, n_tokens, n_labels = get_old_new_tokens_labels(old, new, tokenizer)
                    if r_num<0.33:
                        msk_res = get_keep_span_mask(o_tokens, o_labels, n_tokens, nl_tokens=['<ksm>'], SPECIAL_ID=sp_id)
                        sp_id = msk_res[-1]
                        # print(msk_res[0])
                        # print(msk_res[-1])
                    elif r_num<0.66:
                        msk_res = get_corrupted_pred_v2(o_tokens, n_tokens, nl_tokens=['<dae>'], SPECIAL_ID=sp_id)
                        sp_id = msk_res[-1]
                        # old_new_list.append((msk_res[0][3:], msk_res[1]))
                        # print(msk_res[0])
                        # print(msk_res[-1])
                    else:
                        msk_res = get_target_generation(o_tokens, n_tokens, nl_tokens=['<etg>'], SPECIAL_ID=sp_id)
                        sp_id = msk_res[-1]
                        # old_new_list.append((msk_res[0], msk_res[1]))
                        # print(msk_res[0])
                        # print(msk_res[-1])
                    old_new_list.append((tokenizer.convert_tokens_to_string(msk_res[0][3:]), tokenizer.convert_tokens_to_string(msk_res[1])))
                except:
                    continue
            if len(old_new_list)==0:
                continue
            if len(old_new_list)==1:
                item_list.extend(get_diffusion(old_new_list,item))
            else:
                if len(tokenizer.tokenize(''.join([oni[0] for oni in old_new_list])))<=510 and len(tokenizer.tokenize(''.join([oni[1] for oni in old_new_list])))<=510:
                    item_list.extend(get_diffusion(old_new_list,item))
                else: # >510
                    # split old_new_list
                    diff_list_v2 = [d for d in old_new_list if len(tokenizer.tokenize(d[0]))<=510 and len(tokenizer.tokenize(d[1]))<=510]
                    if len(tokenizer.tokenize(''.join([oni[0] for oni in old_new_list])))<510:
                        item_list.extend(get_diffusion(diff_list_v2,item))
                    else:
                        length_diff = len(diff_list_v2)
                        for j in range(length_diff//2):
                            diff_list_v3 = diff_list_v2[j*2:j*2+2]
                            item_list.extend(get_diffusion(diff_list_v3,item))
        except Exception as e:
                pass
        finally:
            signal.alarm(0)
        # item_list.append(old_new_list)
        # break
        # print("item_list: ",item_list)
        if item_list:
            df = pd.DataFrame(item_list)[['commit','repos', 'msg', 'old', 'new', 'source_track']]
            for i in range(len(df)):
                item = df.iloc[i,:].to_dict()
                file.write(json.dumps(item)+'\n')


