import argparse
import os
import random
import sys
import numpy as np
sys.path.append(os.getcwd())
sys.path.append('..')
import torch
import multiprocessing
from transformers.utils import logging
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import torch.distributed as dist
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from src.models import build_or_load_gen_model
from src.configs import add_args
from tqdm import tqdm
import time

# import deepspeed
from accelerate import Accelerator
from accelerate.state import AcceleratorState


from torch.utils.data import Dataset
from transformers import RobertaTokenizer, T5Tokenizer
import torch
import logging
import os
import time
import random
from copy import deepcopy
import json
import re
from tqdm import tqdm

import diff_utils
import code_tokenize_utils as ctu
import mask_diff_utils as mdu

logger = logging.getLogger(__name__)

class TextDataset(Dataset):

    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, random_sample_num=-1):
        world_size = get_world_size()
        local_rank = get_rank()

        self.cnt = 0
        self.tokenizer = tokenizer
        self.args = args
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"

        savep = file_path.replace(".jsonl", "_"+tokenizer_type+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank) + ".exps")

        if os.path.exists(savep):
            logger.warning("Loading examples from {}".format(savep))
            try:
                self.feats = torch.load(savep)
            except Exception as e:
                logger.warning(f"Loading Failed")
                raise(e)
        else:
            logger.warning("Reading examples from {}".format(file_path))
            start = time.time()
            try:
                logger.warning(f"****reading file")
                examples = read_examples(args, file_path, samplenum, tokenizer=tokenizer)
                logger.warning(f"****reading finish")
            except Exception as e:
                logger.warning(f"****reading fail")
                logger.warning(f"Error in reading examples: {e}")
            end = time.time()
            logger.warning(f"Read examples time cost: {end-start}")
            logger.warning(f"Length of examples:  {len(examples)}")
            if random_sample_num != -1 and examples.__len__() > random_sample_num:
                examples = random.sample(examples, random_sample_num)
            length = len(examples)
            print("******* Length of examples******* : ", length)

            feat_list = []
            for idx, example in tqdm(zip(range(len(examples)),examples), total=len(examples)):
                feat = self.convert_examples_to_features_v2([example, tokenizer, args])
                if feat:
                    feat_list.extend(feat)
                if args.debug and idx==0:
                    print("example of msg: ", example.msg)
                    print("example of nltokens: ", example.nl_tokens)
            print("******* length of all feats ******* : ", len(feat_list))

            # featss = pool.map(self.convert_examples_to_features_v2,
            #                    [(example, tokenizer, args) for example in examples])
            # # expand the lists
            # feat_list = [feat for feats in featss for feat in feats]
            # print("******* length of all feats ******* : ", len(feat_list))

            self.feats = []
            if len(feat_list)%world_size!=0:
                feat_list = feat_list[:-(len(feat_list)%world_size)]
                print(len(feat_list), -(len(feat_list)%world_size))
            for idx, fs in tqdm(zip(range(len(feat_list)),feat_list), total=len(feat_list)):
                # multi gpu
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
                if idx % world_size != local_rank:
                    continue
                self.feats.append(fs)
            print(f"******* length of feats in rank {local_rank} ******* : ", len(self.feats))
            torch.save(self.feats, savep)

        logger.warning("Loading Finished!")
        if args.debug:
            logger.warning(f"Examples feats size: {self.feats.__len__()}")

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def convert_examples_to_features_v2(self, item):
        example, tokenizer, args = item
        o_tokens, o_labels, n_tokens, nl_tokens = example.o_tokens, example.o_labels, example.n_tokens,  example.nl_tokens
        old = example.old
        new = example.new
        dtype = example.dtype
        
        if dtype != "commitpack_diffusion":
            if len(example.nl_tokens) > 0: #masked example
                exs = []
                try:
                    if random.random() < 0.33:
                        msk_res = mdu.get_keep_span_mask(o_tokens, o_labels, n_tokens, ctype="<"+example.lang.strip()+">", nl_tokens=nl_tokens)
                        input_type = '<ksm>'
                        output_type = '<updated_code>'

                    elif random.random() < 0.66:
                        msk_res = mdu.get_corrupted_pred(old, new, ctype="<"+example.lang.strip()+">", nl_tokens=nl_tokens)
                        input_type = '<ade>'
                        output_type = '<updated_code>'
                    
                    else:
                        msk_res = mdu.get_target_generation(o_tokens, n_tokens, ctype="<"+example.lang.strip()+">", nl_tokens=nl_tokens)
                        input_type = '<etg>'
                        output_type = '<updated_code>'

                    if msk_res:
                        source_ids, input_labels, target_ids = self.tokenize(args, tokenizer, [input_type]+msk_res[0], [output_type]+msk_res[1])
                        feats =  ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="masked")
                        exs.append(feats)
                except Exception as e:
                    if args.debug:
                        print("mask error: ",e)
                return exs
            else:
                return []
            
        else: #commitpack_diffusion
            exs = []
            try:
                msk_res = [['<msg>'] + nl_tokens + ["<"+example.lang.strip()+">"] + o_tokens, n_tokens]
                input_type = '<dif>'
                output_type = '<updated_code>'
                if msk_res:
                    source_ids, input_labels, target_ids = self.tokenize(args, tokenizer, [input_type]+msk_res[0], [output_type]+msk_res[1])
                    feats =  ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="commitpack_diffusion")
                    exs.append(feats)
            except Exception as e:
                if args.debug:
                    print("mask error: ",e)
            return exs

    def tokenize(self, args, tokenizer, source, target, type=None):
        # if args.debug:
        #     print("source: ", ' '.join(source))
        #     print("target: ", ' '.join(target))
        source_ids = tokenizer.encode(
                    ' '.join(source), max_length=args.max_source_length - 2, truncation=True, padding='max_length')
        target_ids = tokenizer.encode(
                    ' '.join(target), max_length=args.max_target_length - 2, truncation=True, padding='max_length')
        input_labels = [-100] * args.max_source_length
        return  source_ids, input_labels, target_ids

import pandas as pd
def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)
def read_examples(args, filename, data_num=-1, tokenizer=None):
    """Read examples from filename."""
    examples = []
    idx = 0
    print(f"Loading from {filename}")
    data = read_data(filename)
    # with open(filename) as f:
    for _, js in tqdm(data.iterrows(), total=len(data)):
        dtype = js["type"] # ccn, commitpack, jcp
        maxl = args.max_source_length
        try:
            example = ReviewExample(idx=idx, old=js["old"], new=js["new"], msg=js["nl"], max_len=maxl,
                    max_tgt_len=args.max_target_length, lang=js["lang"], tokenizer=tokenizer, dtype=dtype)

            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                idx += 1
                if idx == data_num:
                    break
        except:
            idx += 1
            if idx == data_num:
                break
    return examples

class ReviewExample(object):
    """A single training/test example."""

    def __init__(self, idx, old, new, msg, max_len, max_tgt_len, lang, tokenizer, dtype, skip_unavail=True):
        self.idx = idx      # idx is useless yet
        self.msg = msg
        self.max_len = max_len
        self.old = old
        self.new = new
        self.o_tokens = []
        self.n_tokens = []
        self.o_labels = []
        self.n_labels = []
        self.nl_tokens = []
        self.avail = True
        self.lang = lang
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.postprocess()

    def postprocess(self):
        if self.old != self.new and self.old.strip() and self.new.strip():
            try:
                self.o_tokens, self.o_labels, self.n_tokens, self.n_labels = self.get_old_new_tokens_labels(self.old, self.new)
                if self.msg != '':
                    self.nl_tokens = ctu.subtokenize_nl(self.msg, self.tokenizer)
            except Exception as e:
                self.avail = False
                print("Error: ",e)
        else:
            # print("old: ", self.old)
            # print("new: ", self.new)
            self.avail = False
        
    def get_old_new_tokens_labels(self, old, new):
        # ex_token_diff = diff_utils.compute_code_diffs(ctu.get_tokenstr_list(ctu.tokenize_code(old)), ctu.get_tokenstr_list(ctu.tokenize_code(new)))[1]
        ex_token_diff = diff_utils.compute_code_diffs(ctu.subtokenize_code(old, self.tokenizer), ctu.subtokenize_code(new, self.tokenizer))[1]
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


class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, source_labels, target_ids, type):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_labels = source_labels
        self.target_ids = target_ids
        self.type = type


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
def get_rank():
    return int(os.environ.get("RANK", "0"))
def get_world_size():
    return os.environ.get("CUDA_VISIBLE_DEVICES","0").count(',')+1
def find_langs_in_data_dir(data_dir):
    return list(set(
        ["_".join(f[:-6].split("_")[:-1])
         for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    ))
def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)

def num_parameters(model):
    model_parameters = model.parameters()
    return sum([np.prod(p.size()) for p in model_parameters])

def load_scheduler_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    return scheduler, optimizer


def get_loaders(data_list, args, tokenizer, pool):
    def fn(features):
        return features
    # world_size = 1
    # assert len(data_list) > 0, "Empty datalist."
    # each_len = len(data_list) // world_size
    # data_list = data_list[: each_len * world_size]
    random.shuffle(data_list)       # this will shuffle data chunks
    for data_file in data_list:
        logger.warning(f"Start data files {data_file}.")
        # add concat dataset
        datasets = [TextDataset(tokenizer, pool, args, data_file)]

        dataset = ConcatDataset(datasets)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=args.cpu_count, collate_fn=fn, drop_last=True)
        logger.warning(f"Finish data files {data_file}.")
        yield dataset, sampler, dataloader

def save_model(model, optimizer, scheduler, output_dir, config, accelerator):
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    # accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save({
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(), # optimizer is an AcceleratedOptimizer object
        "scheduler": scheduler.state_dict()
    }, output_model_file)
    config.save_pretrained(output_dir)

def pretraining(args, config, model, tokenizer, scheduler, optimizer, accelerator):
    not_loss_dec_cnt, global_step, best_ppl = 0, 0, 1e6
    args.train_batch_size = int(args.train_batch_size)
    args.cpu_count = int(multiprocessing.cpu_count()//torch.cuda.device_count())
    pool = multiprocessing.Pool(args.cpu_count//2//get_world_size())
    data_list =[os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if x.startswith("train") and x.endswith(".jsonl")]

    for epoch in range(1, args.num_train_epochs + 1):
        save_seed = args.seed
        args.seed += epoch
        set_seed(args.seed)
        args.seed = save_seed
        random.shuffle(data_list)

        # WARNING: this is a iterator, to save memory
        data_tuples = get_loaders(data_list, args, tokenizer, pool)
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for dataset, sampler, dataloader in data_tuples:
            model.train()
            bar = tqdm(dataloader, total=len(dataloader), desc="Training")
            for step, examples in enumerate(bar, 1):
                source_ids = torch.tensor(
                    [ex.source_ids for ex in examples], dtype=torch.long
                ).to(accelerator.device)
                source_labels = torch.tensor(
                    [ex.source_labels for ex in examples], dtype=torch.long
                ).to(accelerator.device)
                target_ids = torch.tensor(
                    [ex.target_ids for ex in examples], dtype=torch.long
                ).to(accelerator.device)

                source_mask = source_ids.ne(tokenizer.pad_id)
                target_mask = target_ids.ne(tokenizer.pad_id)

                outputs = model(
                    input_ids=source_ids,
                    labels=target_ids,
                    attention_mask=source_mask,
                    decoder_attention_mask=target_mask
                )
                loss = outputs.loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                # loss.backward()

                # accelerator
                # accelerator.backward(loss)
                
                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    accelerator.backward(loss)
                   
                    global_step += 1
                    if is_rank_0() and global_step % args.log_steps == 0:
                        train_loss = round(
                            tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                            4,
                        )
                        # bar.set_description("Step {}/{}: Epoch {}: Train loss {}".format(global_step, args.train_steps, epoch, round(train_loss, 3)))
                        bar.set_description("Step {}/{}: Epoch {}:  Rank {}: Train loss {}".format(global_step, args.train_steps, epoch, get_rank(), round(train_loss, 3)))
                else:
                    with accelerator.no_sync(model):
                        accelerator.backward(loss)

                if accelerator.is_main_process and global_step % args.save_steps == 0 and global_step>0:
                    output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step))
                    save_model(model, optimizer, scheduler, output_dir, config, accelerator)
                    accelerator.print( "Save the {}-step model and optimizer into {}".format(global_step, output_dir))
                    
    # reach max epochs, not max steps
    # if is_rank_0():
    # end training
    if accelerator.is_main_process:
        # accelerator.save_state(output_dir)
        # config.save_pretrained(output_dir)
        output_dir = os.path.join(args.output_dir, "checkpoints-last")
        save_model(model, optimizer, scheduler, output_dir, config, accelerator)
    accelerator.print(f"Reach max steps {args.train_steps}.")
    return


def main(args):
    # accelerator
    accelerator = Accelerator()

    config, model, tokenizer = build_or_load_gen_model(args)
    logger.info(f"Starting Training from {args.model_type}")
    logger.info(f"Total parameters : {num_parameters(model)}")
    
    if not os.path.exists("{}/checkpoints-last".format(args.output_dir)):
        os.makedirs("{}/checkpoints-last".format(args.output_dir))
        
    if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)):
        model = accelerator.unwrap_model(model)
        model.load_state_dict(torch.load("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))['model'])
        accelerator.print("Load model from {}/checkpoints-last/pytorch_model.bin".format(args.output_dir))

    # args.train_steps = 150000
    scheduler, optimizer = load_scheduler_optimizer(args, model)
    accelerator.print("Prepare data and get training steps finished.")

    # prepare
    # When using DeepSpeed `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders 
    # or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=args.train_batch_size
    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]=args.gradient_accumulation_steps
    model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
    # training
    model = model.to(accelerator.device)

    pretraining(args, config, model, tokenizer, scheduler, optimizer, accelerator)


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_seed(args.seed)
    logger.info(args)
    main(args)
    logger.warning("Pre-training finished.")
    logger.warning("Finish training and take %s", get_elapse_time(t0))
