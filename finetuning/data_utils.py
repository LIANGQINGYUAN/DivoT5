import os
import json
import pandas as pd

def read_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split=False):
# def read_examples(filename, data_num):
    """Read examples from filename."""
    if "JITCommentUpdate" in data_dir:
        examples = read_comment_update_examples(data_dir, data_num, world_size, local_rank, is_split)
    elif "code-refinement" in data_dir:
        examples = read_refinement_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    elif 'code-review' in data_dir:
        examples = read_codereview_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    elif 'comment-update' in data_dir:
        examples = read_comment_update_examples_v2(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    elif 'CodeRefinement' in data_dir:
        examples = read_CodeRefinement_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    elif 'CUAAAI21' in data_dir:
        examples = read_cuaaai21(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    else:
        examples = read_buggy_fixed(data_dir, split_tag, data_num, world_size, local_rank, is_split)
    return examples

def read_cuaaai21(data_dir, split_tag, data_num, world_size, local_rank, is_split):
    examples = []
    if split_tag == 'train':
        filename = os.path.join(data_dir, f"train.json")
    elif split_tag == 'test':
        filename = os.path.join(data_dir, f"test.json")
    else:
        filename = os.path.join(data_dir, f"valid.json")
    data = pd.read_json(filename)
    for idx in range(len(data)):
        x= data.iloc[idx].to_dict()
        examples.append(
            Example(
                idx=idx,
                source= ' '.join((x["old_comment_raw"].strip()+ " <s> " + x["old_code_raw"].strip()+ " <s> " + x["new_code_raw"].strip().replace("\n"," ")).split(" ")),
                target= ' '.join(x["new_comment_raw"].replace("\n"," ").strip().split(" ")),
                old_comm=' '.join(x["old_comment_raw"].replace("\n"," ").strip().split(" "))
            )
        )
    return examples

def read_buggy_fixed(data_dir, split_tag, data_num, world_size, local_rank, is_split):
    examples = []
    file_path = data_dir
    file_tag = file_path[file_path.rindex("/")+1:]
    if 'train' == split_tag:
        src_file = os.path.join(file_path, f'train.{file_tag}.buggy')
        tgt_file = os.path.join(file_path, f'train.{file_tag}.fixed')
    elif 'test' == split_tag:
        src_file = os.path.join(file_path, f'test.{file_tag}.buggy')
        tgt_file = os.path.join(file_path, f'test.{file_tag}.fixed')
    else:
        src_file = os.path.join(file_path, f'valid.{file_tag}.buggy')
        tgt_file = os.path.join(file_path, f'valid.{file_tag}.fixed')
    examples=[]
    srcs = open(src_file).readlines()
    tgts = open(tgt_file).readlines()
    for idx, s, t in zip(range(len(srcs)), srcs, tgts):
        examples.append(
                Example(
                        idx = idx,
                        # source="<msg> " + s.split('<s>')[1] + " <java> " + s.split('<s>')[0],
                        source = " ".join(s.split()),
                        target = " ".join(t.split()),
                        ) 
            )
    return examples
    return examples

def read_comment_update_examples_v2(data_dir, split_tag, data_num, world_size, local_rank, is_split=False):
    examples = []
    if split_tag == 'train':
        filename = os.path.join(data_dir, f"train.json")
    elif split_tag == 'test':
        filename = os.path.join(data_dir, f"test.json")
    else:
        filename = os.path.join(data_dir, f"valid.json")
    data = pd.read_json(filename)
    for idx in range(len(data)): #(data.old_comment.tolist(), data.old_code.tolist(), data.new_comm.tolist(), data.new_code.tolist()):
        x= data.iloc[idx].to_dict()
        examples.append(
            Example(
                idx=idx,
                source= ' '.join((x["old_comment"].strip()+ " <s> " + x["old_code"].strip()+ " <s> "+x["new_code"].strip().replace("\n"," ")).split(" ")),
                target= ' '.join(x["new_comment"].replace("\n"," ").strip().split(" ")),
                old_comm=' '.join(x["old_comment"].replace("\n"," ").strip().split(" "))
            )
        )
    return examples

def read_refinement_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split=False):
    examples = []
    file_path = data_dir
    if 'train' == split_tag:
        src_file = os.path.join(file_path, 'train.buggy-fixed.buggy')
        tgt_file = os.path.join(file_path, 'train.buggy-fixed.fixed')
    elif 'test' == split_tag:
        src_file = os.path.join(file_path, 'test.buggy-fixed.buggy')
        tgt_file = os.path.join(file_path, 'test.buggy-fixed.fixed')
    else:
        src_file = os.path.join(file_path, 'valid.buggy-fixed.buggy')
        tgt_file = os.path.join(file_path, 'valid.buggy-fixed.fixed')
    examples=[]
    srcs = open(src_file).readlines()
    tgts = open(tgt_file).readlines()
    for idx, s, t in zip(range(len(srcs)), srcs, tgts):
        examples.append(
                Example(
                        idx = idx,
                        source=" ".join(s.split()),
                        target = " ".join(t.split()),
                        ) 
            )
    return examples

def read_codereview_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split=False):
    examples = []
    file_path = data_dir
    if 'train' == split_tag:
        src_file = os.path.join(file_path, 'train.code-review.buggy')
        tgt_file = os.path.join(file_path, 'train.code-review.fixed')
    elif 'test' == split_tag:
        src_file = os.path.join(file_path, 'test.code-review.buggy')
        tgt_file = os.path.join(file_path, 'test.code-review.fixed')
    else:
        src_file = os.path.join(file_path, 'valid.code-review.buggy')
        tgt_file = os.path.join(file_path, 'valid.code-review.fixed')
    examples=[]
    srcs = open(src_file,encoding="utf-8").readlines()
    tgts = open(tgt_file,encoding="utf-8").readlines()
    for idx, s, t in zip(range(len(srcs)), srcs, tgts):
        examples.append(
                Example(
                        idx = idx,
                        # source="<msg> " + s.split('<s>')[1] + " <java> " + s.split('<s>')[0],
                        source = " ".join(s.split()),
                        target = " ".join(t.split()),
                        ) 
            )
    return examples

def get_old_new_from_diff(diff):
    def remove_blank_lines(code):
        return'\n'.join([line for line in code.split('\n') if  len(line.replace(' ','')) != 0])
    matchres = diff[diff.index('@@\n\n')+len('@@\n\n'):]
    diff = matchres
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
    return old, new 

def read_comment_update_examples(filename, data_num, world_size, local_rank, is_split=False):
    examples = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            if is_split and idx % world_size != local_rank:
                continue
            x = json.loads(line)
            old, new = get_old_new_from_diff(x['diff'])
            examples.append(
                Example(
                    idx=idx,
                    source= x["old_nl"].strip()+ " <s> " + old.strip() + " <s> " + new.strip().replace("\n"," "),
                    target= ' '.join(x["nl"].replace("\n"," ").strip().split(" ")),
                    old_comm=' '.join(x["old_nl"].replace("\n"," ").strip().split(" "))
                )
            )
            idx += 1
            
            if idx == data_num:
                break
    return examples

def read_CodeRefinement_examples(data_dir, split_tag, data_num, world_size, local_rank, is_split):
    examples = []
    if split_tag == 'train':
        filename = os.path.join(data_dir, f"train.json")
    elif split_tag == 'test':
        filename = os.path.join(data_dir, f"test.json")
    else:
        filename = os.path.join(data_dir, f"valid.json")
    with open(filename) as f:
        for idx, line in enumerate(f):
            if is_split and idx % world_size != local_rank:
                continue
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source= x["old"]+ " <s> " + x["comment"],
                    target= x["new"]
                )
            )
            idx += 1
            
            if idx == data_num:
                break 
    return examples

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 old_comm='',
                 old_code='',
                 new_comm='',
                 new_code=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.old_comm = old_comm
        self.old_code = old_code
        self.new_comm = new_comm
        self.new_code = new_code

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    if args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_len, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_len, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
    )