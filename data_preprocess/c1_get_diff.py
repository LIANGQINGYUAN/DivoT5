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

import difflib

def split_diff(diff_output):
    diff_segments = []
    current_segment = []

    for line in diff_output:
        if line.startswith("@@"):
            # start new diff
            if current_segment:
                diff_segments.append(current_segment)
            current_segment = [line]
        else:
            current_segment.append(line)

    # add the last one
    if current_segment:
        diff_segments.append(current_segment)

    return diff_segments

import re
def remove_java_comments(source_code):
    # regex
    pattern = r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)'
    # remove comments
    cleaned_code = re.sub(pattern, '', source_code, flags=re.MULTILINE)
    code = "\n".join([c for c in cleaned_code.splitlines() if len(c.split())!=0])
    return code

def calculate_diff(str1, str2):
    # str1 = remove_java_comments(str1)
    # str2 = remove_java_comments(str2)

    # get diff
    diff_output = difflib.unified_diff(str1.splitlines(), str2.splitlines(), lineterm='')

    # split diff
    diff_segments = split_diff(diff_output)
    return ['\n'.join(ccdiff[1:]) for ccdiff in diff_segments][1:]

import os 
files = os.listdir('../json/')

files = sorted(files, key=lambda x: int(x.split('-')[1].split('.')[0]))


import signal
def handler(signum, frame):
    raise Exception("Time's up!")
signal.signal(signal.SIGALRM, handler)

base_dir = '../json/'
data_list = []
with open('./CommitPack_Diff_Java_V2.json', 'w') as wfile:
    for f in tqdm(files):
        item = read_data(base_dir+f)
        # data_list.append(item)
        for i, row in tqdm(item.iterrows(), total=len(item)):
            signal.alarm(5)
            try:
                diff = calculate_diff(row['old_contents'], row['new_contents'])
            except Exception as e:
                pass
            finally:
                signal.alarm(0)
            
            if len(diff)>0:
                diff = '\n<@@DiffSegmentsSplit@@>\n'.join(diff) if len(diff)>1 else diff[0]
                wfile.write(json.dumps({'commit':row['commit'], "repos":row['repos'], 'msg':row['message'], 'diff':diff})+'\n')
