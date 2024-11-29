import json
import pandas as pd
from tqdm import tqdm
import sys
import re 
import code_tokenize_utils as ctu
from mask import span_mask
import numpy as np
import diff_utils
import random
import javalang

def insert_token(tokens, index, new_token):
    tokens.insert(index, new_token)

def delete_token(tokens, index):
    del tokens[index]

def replace_token(tokens, index, new_token):
    tokens[index] = new_token

def corrupt_code(original_code, insert_prob, delete_prob, replace_prob):
    tokens = list(javalang.tokenizer.tokenize(original_code))
    SPECIAL_ID = 0
    for index, token in enumerate(tokens):
        rand = random.uniform(0, 1)

        if rand < insert_prob[1] and rand > insert_prob[0] and SPECIAL_ID<99:
            new_token = javalang.tokenizer.Identifier(f"<extra_id_{SPECIAL_ID}>", position=(0, 0))
            insert_token(tokens, index, new_token)
            SPECIAL_ID+=1
            continue
        
        if rand < delete_prob[1] and rand > delete_prob[0]:
            delete_token(tokens, index)
            continue

        if rand < insert_prob[1] and rand > insert_prob[0] and SPECIAL_ID<99:
            new_token = javalang.tokenizer.Identifier(f"<extra_id_{SPECIAL_ID}>", position=(0, 0))
            replace_token(tokens, index, new_token)
            SPECIAL_ID+=1
            continue

    return tokens

def tokenize_and_mark_identifiers(java_code):
    tokens = list(javalang.tokenizer.tokenize(java_code))
    marked_tokens = []

    for token in tokens:
        if isinstance(token, javalang.tokenizer.Keyword):
            marked_tokens.append((token.value, False))  # Non-Identifier token (Keyword)
        elif isinstance(token, javalang.tokenizer.Identifier):
            marked_tokens.append((token.value, True))   # Identifier token
        else:
            marked_tokens.append((token.value, False))  # Non-Identifier token

    return marked_tokens

def get_old_new_from_diff(diff):
    def remove_blank_lines(code):
        return'\n'.join([line for line in code.split('\n') if  len(line.replace(' ','')) != 0])
    regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
    matchres = re.match(regex, diff)
    if matchres:
        diff = diff[matchres.span()[1]:]
        old = []
        new = []
        for line in diff.split('\n'):
            if line.startswith('-'):
                old.append(line[1:])
            elif line.startswith('+'):
                new.append(line[1:])
            elif line.startswith(' '):
                old.append(line[1:])
                new.append(line[1:])
            else:
                old.append(line)
                new.append(line)
        old = remove_blank_lines('\n'.join(old))
        new = remove_blank_lines('\n'.join(new))
        if old != new:
            return old, new 
        else:
            return None, None
    else:
        return None, None

def get_old_new_tokens_labels(old, new):
    ex_token_diff = diff_utils.compute_code_diffs(ctu.get_tokenstr_list(ctu.tokenize_code(old)), ctu.get_tokenstr_list(ctu.tokenize_code(new)))[1]
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

def get_span_mask_from_mask(mask, tokens):
    SPECIAL_ID = 0
    spmask = []
    for i in range(len(mask)):
        if i==0:
            if mask[i] == '<mask>' and SPECIAL_ID<99:
                spmask.append(f"<extra_id_{SPECIAL_ID}>")
                SPECIAL_ID+=1
            else:
                spmask.append(tokens[i])
        else:
            if mask[i] == '<mask>' and mask[i-1] != mask[i] and SPECIAL_ID<99:
                spmask.append(f"<extra_id_{SPECIAL_ID}>")
                SPECIAL_ID+=1
            elif mask[i] != '<mask>':
                spmask.append(tokens[i])
    return spmask

def get_nl_mask(nl_tokens):
    nl_length = len(nl_tokens)
    if nl_length>5:
        mean_span = 2.5
        mean_sapn_length = max(int(nl_length*0.2//mean_span), 1)
        masktags = span_mask.random_spans_noise_mask(nl_length, mean_span, mean_sapn_length)
        tokens_with_mask = [t if m==0 else '<mask>' for m, t in zip(masktags, nl_tokens)]
        return tokens_with_mask
    return nl_tokens

def get_keep_span_mask(o_tokens, o_labels, n_tokens, ctype='<java>', nl_tokens=None):
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
    tokens_with_mask = get_span_mask_from_mask(tokens_with_mask, o_tokens)
    nl_tokens_mask = get_nl_mask(nl_tokens)
    
    input_mask_tokens = ['<msg>'] + nl_tokens_mask + [ctype] + tokens_with_mask
    input_original_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    input_tokens = get_span_mask_from_mask(input_mask_tokens, input_original_tokens)

    output_labels = n_tokens
    return input_tokens, output_labels

def get_maskid_pred(old, new, ctype='<java>', nl_tokens=None):
    try:
        marked_tokens = tokenize_and_mark_identifiers(old)
        new_tokens = tokenize_and_mark_identifiers(new)
    except:
        return None
    
    SPECIAL_ID = 0
    tokens = []
    for tk, is_identifier in marked_tokens:
        if is_identifier and SPECIAL_ID<99:
            tokens.append(f"<extra_id_{SPECIAL_ID}>")
            SPECIAL_ID+=1
        else:
            tokens.append(tk)
    if nl_tokens:
        input_tokens = ['<msg>'] + nl_tokens + [ctype] + tokens
    else:
        input_tokens = [ctype] + tokens
    output_labels = [ctype] + [t[0] for t in new_tokens]
    return input_tokens, output_labels

def get_corrupted_pred(old, new, ctype='<java>', nl_tokens=None):
    code_delete_prob = (0, 0.495)
    code_insert_prob = (0.495, 0.71)
    code_replace_prob = (0.71, 1)
    try:
        corrupted_tokens = corrupt_code(old, code_insert_prob, code_delete_prob,  code_replace_prob)
        tokens = [token.value for token in corrupted_tokens]
        new_tokens = tokenize_and_mark_identifiers(new)
    except:
        return None
    if nl_tokens:
        input_tokens = ['<msg>'] + nl_tokens + [ctype] + tokens
    else:
        input_tokens = [ctype] + tokens
    output_labels = [ctype] + [t[0] for t in new_tokens]
    return input_tokens, output_labels

def get_edit_tag_mask(o_tokens, o_labels, ctype='<java>', nl_tokens=None):
    length_keep = len(o_labels) - sum(o_labels)
    masks = [random.random() < 0.2 for _ in range(length_keep)]
    masks_change = [random.random() < 0.5 for _ in range(sum(o_labels))]
    # print(masks, masks_change)
    code_tokens = []
    SPECIAL_ID = 0
    output_labels = ''
    index = 0 
    index_change = 0
    for t, l in zip(o_tokens, o_labels):
        if l == 0: #keep
            if masks[index] and SPECIAL_ID<99:
                code_tokens.append(f"<extra_id_{SPECIAL_ID}>")
                code_tokens.append(t)
                output_labels+=f"<extra_id_{SPECIAL_ID}>:<KEEP>, "
                SPECIAL_ID+=1
                index+=1
            else:
                code_tokens.append(t)
                index+=1
        else: #change
            if masks_change[index_change] and SPECIAL_ID<99:
                code_tokens.append(f"<extra_id_{SPECIAL_ID}>")
                code_tokens.append(t)
                output_labels+=f"<extra_id_{SPECIAL_ID}>:<CHG>, "
                SPECIAL_ID+=1
                index_change+=1
            else:
                code_tokens.append(t)
                index_change+=1
    if len(output_labels)>0:
        output_labels = output_labels[:-2] # remove ', '
    # nl_tokens_mask = get_nl_mask(nl_tokens)
    # nl_tokens = get_span_mask_from_mask(nl_tokens_mask, nl_tokens)
    input_tokens = ['<msg>'] + nl_tokens + [ctype] + code_tokens
    return input_tokens, output_labels.split(' ')

def get_edit_content_mask(o_tokens, o_labels, n_tokens, ctype='<java>', nl_tokens=None):
    tokens_with_mask = []
    masks_change = [random.random() < 0.5 for _ in range(sum(o_labels))]
    index_change = 0
    for t, l in zip(o_tokens, o_labels):
        if l!=0 and masks_change[index_change]:
                tokens_with_mask.append('<mask>')
                index_change+=1
        else:
            tokens_with_mask.append(t)
    
    nl_tokens_mask = get_nl_mask(nl_tokens)
    input_mask_tokens = ['<msg>'] + nl_tokens_mask + [ctype] + tokens_with_mask
    input_original_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    input_tokens = get_span_mask_from_mask(input_mask_tokens, input_original_tokens)

    output_labels = n_tokens
    return input_tokens, output_labels

def get_target_generation(o_tokens, n_tokens, ctype='<java>', nl_tokens=None):
    input_tokens = []
    # input_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens

    nl_tokens_mask = get_nl_mask(nl_tokens)
    o_tokens_mask = get_nl_mask(o_tokens)
    input_mask_tokens = ['<msg>'] + nl_tokens_mask + [ctype] + o_tokens_mask
    input_original_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    input_tokens = get_span_mask_from_mask(input_mask_tokens, input_original_tokens)

    output_labels = n_tokens
    return input_tokens, output_labels

def get_edit_paln_generation(o_tokens, n_tokens, ctype='<java>', nl_tokens=None):
    input_tokens = []
    edit_plan = diff_utils.compute_minimal_code_diffs(o_tokens, n_tokens)[0]
    input_tokens = ['<msg>'] + nl_tokens + [ctype] + o_tokens
    output_labels = edit_plan
    return input_tokens, n_tokens + ["<s>"] + output_labels

def get_change_2_msg_generation(o_tokens, n_tokens, ctype='<java>', nl_tokens=None):
    input_tokens = []
    input_tokens = [ctype] + o_tokens[:250] + ['<s>'] + n_tokens[:250]
    output_labels =  ['<msg>'] + nl_tokens
    return input_tokens,  output_labels

def get_edit_paln_2_msg_generation(o_tokens, n_tokens, ctype='<java>', nl_tokens=None):
    input_tokens = []
    edit_plan = diff_utils.compute_minimal_code_diffs(o_tokens, n_tokens)[0]
    input_tokens = edit_plan
    output_labels =  ['<msg>'] + nl_tokens
    return input_tokens,  output_labels


def get_target_generation_no_nl(o_tokens, n_tokens, ctype='<java>'):
    input_tokens = []
    # input_tokens = [ctype] + o_tokens

    o_tokens_mask = get_nl_mask(o_tokens)
    input_mask_tokens = [ctype] + o_tokens_mask
    input_original_tokens = [ctype] + o_tokens
    input_tokens = get_span_mask_from_mask(input_mask_tokens, input_original_tokens)

    output_labels = n_tokens
    return input_tokens, output_labels

def get_edit_paln_generation_no_nl(o_tokens, n_tokens, ctype='<java>'):
    input_tokens = []
    edit_plan = diff_utils.compute_minimal_code_diffs(o_tokens, n_tokens)[0]
    input_tokens = [ctype] + o_tokens
    output_labels = edit_plan
    return input_tokens,  n_tokens + ["<s>"] + output_labels

def get_keep_span_mask_no_nl(o_tokens, o_labels, n_tokens, ctype='<java>', nl_tokens=None):
    length_keep = len(o_labels) - sum(o_labels)
    # print(length_keep)
    mean_span = 2.5
    mean_sapn_length = max(int(length_keep*0.3//mean_span), 1)
    try:
        masktags = span_mask.random_spans_noise_mask(length_keep, mean_span, mean_sapn_length)
    except:
        masktags = [0]*length_keep
    # print("*",length_keep,len(masktags), masktags, masktags)
    # import sys
    # sys.exit()
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
    tokens_with_mask = get_span_mask_from_mask(tokens_with_mask, o_tokens)
    
    input_mask_tokens = [ctype] + tokens_with_mask
    input_original_tokens = [ctype] + o_tokens
    input_tokens = get_span_mask_from_mask(input_mask_tokens, input_original_tokens)

    output_labels = n_tokens
    return input_tokens, output_labels