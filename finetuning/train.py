import os
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed, T5ForConditionalGeneration, RobertaTokenizer

import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append("..")
from evaluator.bleu import _bleu, bleu_from_list
from evaluator.CodeBLEU.calc_code_bleu import get_codebleu

import json
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
from typing import List

from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState

from data_utils import read_examples, convert_examples_to_features

sys.path.append(os.getcwd())
from models import build_or_load_gen_model
from cct5models import build_or_load_gen_model_cct5

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
def get_rank():
    return int(os.environ.get("RANK", "0"))
def get_world_size():
    return os.environ.get("CUDA_VISIBLE_DEVICES","0").count(',')+1
def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')
def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
def save_model(model, optimizer, scheduler, output_dir, accelerator):
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    # accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save({
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(), # optimizer is an AcceleratedOptimizer object
        "scheduler": scheduler.state_dict()
    }, output_model_file)
    
def load_and_cache_gen_data(args, data_dir, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    local_rank = get_rank()
    world_size = get_world_size()
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    
    if split_tag == 'train':
        cache_fn = '{}/{}.exps'.format(args.cache_path, split_tag + "_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank)  + data_tag)
        examples = read_examples(data_dir, split_tag, args.data_num, world_size, local_rank, True)
    else:
        cache_fn = '{}/{}.exps'.format(args.cache_path, split_tag + data_tag)
        examples = read_examples(data_dir, split_tag, args.data_num, world_size, local_rank, False)
    # cache_fn = '{}/{}.exps'.format(args.cache_path, split_tag + data_tag)
    # examples = read_examples(filename, args.data_num)
    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if os.path.exists(cache_fn) and not is_sample:
        # logger.info("Load cache data from %s", cache_fn)
        log_dist(f"Load cache data from {cache_fn}", ranks=[0], level=logging.INFO)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            log_dist(f"Sample 5k data for computing bleu from {data_dir}", ranks=[0], level=logging.INFO)
        else:
            log_dist(f"Create cache data into {cache_fn}", ranks=[0], level=logging.INFO)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_ex_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_ex_ids, all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_ex_ids, all_source_ids, all_target_ids)
        if is_rank_0 and not is_sample:
            os.makedirs(args.cache_path, exist_ok=True)
            torch.save(data, cache_fn)
    return examples, data

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria, accelerator):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size_per_replica)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size_per_replica,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size_per_replica)
    
    model.eval()
    pred_ids = []
    eval_exs = []
    bleu, codebleu = 0.0, 0.0
    for batch, exs in tqdm(zip(eval_dataloader, eval_examples), total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[1].to(accelerator.device)
        # source_ids = batch[0]
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if split_tag == 'test':
                preds = model.generate(source_ids,
                    attention_mask=source_mask, use_cache=True, num_beams=args.beam_size,
                    max_length=args.max_target_len)
            else:
                preds = model.generate(source_ids,
                    attention_mask=source_mask, use_cache=True, num_beams=1,
                    # attention_mask=source_mask, use_cache=True,
                    max_length=args.max_target_len)
            
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    args.res_dir = os.path.join(args.output_dir, "prediction")
    os.makedirs(args.res_dir, exist_ok=True)
    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    
    dev_accs, predictions = [], []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            dev_accs.append(pred_nl.strip() == gold.target.strip())
            f.write(json.dumps({"pred":pred_nl.strip()})+ '\n')
            f1.write(json.dumps({"gold":gold.target.strip()}) + '\n')
            f2.write(json.dumps({"source":gold.source.strip()}) + '\n')
    if gold.target.strip():
        # bleu = round(_bleu(gold_fn, output_fn), 2)
        bleu = round(bleu_from_list([g.target.strip() for g in eval_examples], [p.strip() for p in pred_nls]),2)
        if 'J2C' in args.task:
            print('Get CodeBLEU of C# programming language')
            codebleu = round(get_codebleu([g.target.strip() for g in eval_examples], [p.strip() for p in pred_nls], 'c_sharp')* 100,2)
        else:
            print('Get CodeBLEU of Java programming language')
            codebleu = round(get_codebleu([g.target.strip() for g in eval_examples], [p.strip() for p in pred_nls], 'java')* 100,2)
    else:
        bleu = -1
    result = {'em': sum(dev_accs)/len(dev_accs) * 100, 'bleu': bleu, 'codebleu':codebleu}
    accelerator.print("***** Eval results *****")
    for key in sorted(result.keys()):
        accelerator.print(f"  {key} = {str(round(result[key], 4))}")
    return result

def train(args, model, tokenizer):
    # prepare
    t0 = time.time()
    set_seed(args.seed)
    f_summary = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    cpu_cont = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_cont)

    # Initialize accelerator
    accelerator = Accelerator()
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=args.batch_size_per_replica
    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]=args.grad_acc_steps
    model = model.to(accelerator.device)

    if args.do_train:
        # dataloader
        train_examples, train_data = load_and_cache_gen_data(args, args.data_dir, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size_per_replica,
                                        num_workers=4, pin_memory=True)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        num_train_optimization_steps = args.epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.lr_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        # training
        dev_dataset = {}
        training_loss_log = {}
        global_step, best_bleu_em = 0, -1
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        for cur_epoch in range(int(args.epochs)):
            model.train()
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            for step, batch in enumerate(bar):
                batch = tuple(t.to(accelerator.device) for t in batch)
                exi_ids, source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                tr_loss += loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                # loss.backward()
                # model.backward(loss)

                if nb_tr_steps % args.grad_acc_steps == 0:
                    accelerator.backward(loss)
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    # model.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.grad_acc_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                else:
                    with accelerator.no_sync(model):
                        accelerator.backward(loss)
                
            if args.do_eval:
                if 'dev_loss' in dev_dataset:
                    val_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.data_dir, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.data_dir, pool, tokenizer, 'dev',
                                                                        only_src=True, is_sample=False)
                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch, accelerator)
                    dev_bleu, dev_em, dev_codebleu = result['bleu'], result['em'], result['codebleu']
                    # dev_bleu, dev_em = result['gleu'], result['em']
                    dev_bleu_em = dev_bleu + dev_codebleu + dev_em
                    # dev_bleu_em = dev_bleu
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        log_dist("  [%d] Best bleu+em+cbleu: %.2f (bleu: %.2f, em: %.2f, cbleu: %.2f)"%(cur_epoch, dev_bleu_em, dev_bleu, dev_em, dev_codebleu), ranks=[0], level=logging.INFO)
                        log_dist("  " + "*" * 20, ranks=[0], level=logging.INFO)
                        best_bleu_em = dev_bleu_em
                        if is_rank_0():
                            f_summary.write("[%d] Best bleu+em+cbleu changed into %.2f (bleu: %.2f, em: %.2f, cbleu: %.2f)\n" % (cur_epoch, best_bleu_em, dev_bleu, dev_em, dev_codebleu))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir) and is_rank_0:
                            os.makedirs(output_dir)

                        # save model
                        save_model(model, optimizer, scheduler, output_dir, accelerator)

                    else:
                        not_bleu_em_inc_cnt += 1
                        log_dist(f"Bleu does not increase for {not_bleu_em_inc_cnt} epochs", ranks=[0], level=logging.INFO)
                        if is_rank_0():
                            f_summary.write(
                                "[%d] Best bleu+em+cbleu (%.2f) does not drop changed for %d epochs, cur bleu+em+cbleu: %.2f (bleu: %.2f, em: %.2f, cbleu: %.2f)\n" % (
                                    cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em, dev_codebleu))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            log_dist(stop_early_str, ranks=[0], level=logging.INFO)
                            if is_rank_0():f_summary.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        logger.info("Finish training and take %s", get_elapse_time(t0))
        
    if args.do_test and is_rank_0():
        log_dist("  " + "***** Testing *****", ranks=[0], level=logging.INFO)
        log_dist(f"  Batch size = {args.batch_size_per_replica}", ranks=[0], level=logging.INFO)

        for criteria in ['best-bleu']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            log_dist(f"Reload model from {file}", ranks=[0], level=logging.INFO)
            eval_examples, eval_data = load_and_cache_gen_data(args, args.data_dir, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            # load model
            model = accelerator.unwrap_model(model)
            model.load_state_dict(torch.load(file)['model'])
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria, accelerator)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.2f, codebleu: %.2f\n" % (criteria, test_bleu, test_em, test_codebleu)
            # logger.info(result_str)
            log_dist(f"{result_str}", ranks=[0], level=logging.INFO)
            if is_rank_0(): f_summary.write(result_str)

    log_dist(f"Finish and take {get_elapse_time(t0)}", ranks=[0], level=logging.INFO)
    if is_rank_0():
        f_summary.write("Finish and take {}".format(get_elapse_time(t0)))
        f_summary.close()

def main(args):
    argsdict = vars(args)
    if is_rank_0():
        print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.output_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    if 'evocoder' in args.load.lower():
        print("Load EvoCoder model!")
        args.model_type = 'codet5'
        # args.model_name_or_path = 'Salesforce/codet5-base'
        # args.tokenizer_name = 'Salesforce/codet5-base'
        if 'small' in args.load.lower() or 'small' in args.output_dir.lower() or 'wo-' in args.load.lower():
            args.model_name_or_path = 'Salesforce/codet5-small'
            args.tokenizer_name = 'Salesforce/codet5-small'
        else:
            args.model_name_or_path = 'Salesforce/codet5-base'
            args.tokenizer_name = 'Salesforce/codet5-base'
        config, model, tokenizer = build_or_load_gen_model(args)
        # load state
        loading_path = args.load
        # model = accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(loading_path)['model'])
        logger.info("Load model from {}".format(loading_path))

    # codereviewer: microsoft/codereviewer
    elif 'codereviewer' in args.load or 'CoditT5' in args.load:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        logger.info("Load model from {}".format(args.load))
    
    elif 'codet5-small' in args.load or 'codet5-base' in args.load:
        logger.info("Load model from {}".format(args.load))
        model = T5ForConditionalGeneration.from_pretrained(args.load)
        tokenizer = RobertaTokenizer.from_pretrained(args.load)
        
    elif 'CCT5' in args.load:
        args.model_type = 'codet5_CC'
        args.model_name_or_path = 'Salesforce/codet5-base'
        args.tokenizer_name = 'Salesforce/codet5-base'
        config, model, tokenizer = build_or_load_gen_model_cct5(args)
        model.load_state_dict( torch.load(args.load))
        logger.info("Load model from {}".format(args.load))

    if is_rank_0(): print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    train(args, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', default=-1, type=int)
    parser.add_argument('--max_source_len', default=320, type=int)
    parser.add_argument('--max_target_len', default=128, type=int)
    parser.add_argument('--cache_data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_warmup_steps', default=500, type=int)
    parser.add_argument('--batch_size_per_replica', default=10, type=int)
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)

    # Logging and stuff
    parser.add_argument('--output_dir', default="saved_models/summarize_python", type=str)

    # costum training
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_eval', default=True, action='store_true')
    parser.add_argument('--do_test', default=True, action='store_true')
    parser.add_argument('--do_eval_bleu', default=True, action='store_true')
    parser.add_argument('--data_dir', default=None, type=str)
    # parser.add_argument('--train_filename', default=None, type=str)
    # parser.add_argument('--dev_filename', default=None, type=str)
    # parser.add_argument('--test_filename', default=None, type=str)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--always_save_model', default=False, action='store_true')
    parser.add_argument("--task", type=str, required=False)
    parser.add_argument("--add_task_prefix", default=False, action='store_true')

    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--model_type", default="codet5", type=str)
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)