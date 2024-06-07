import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer, RobertaConfig, RobertaModel,
                          RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Model, T5Tokenizer)
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def enrich_vocab(
    args,
    tokenizer,
    config,
    load_extra_ids=True
):
    add_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    
    add_tokens += ['<ksm>', '<ade>', '<etg>', '<updated_code>', '<dif>']

    add_token_ids = [
        tok for tok in add_tokens if tok not in tokenizer.get_vocab()]
    if add_token_ids:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": add_token_ids}
        )

    if load_extra_ids is True:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<extra_id_{}>".format(i) for i in range(99, -1, -1)
                ]
            }
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<e{}>".format(i) for i in range(99, -1, -1)
                ]
            }
        )

        tokenizer.add_special_tokens({"additional_special_tokens": ["<msg>"]})

        langs = [
            "<en>",
            "<python>",
            "<java>",
            "<javascript>",
            "<ruby>",
            "<php>",
            "<go>",
            "<c>",
            "<c_sharp>",
            "<c_plus_plus>",
        ]

    if args.add_lang_ids:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": langs
            }
        )
        config.lang_id = {
            lang: tokenizer.get_vocab()[lang] for lang in langs
        }

    config.vocab_size = len(tokenizer)
    config.bos_token_id = tokenizer.get_vocab()["<s>"]
    config.pad_token_id = tokenizer.get_vocab()["<pad>"]
    config.eos_token_id = tokenizer.get_vocab()["</s>"]
    config.mask_token_id = tokenizer.get_vocab()["<mask>"]
    config.lang_tokens = langs

    tokenizer.special_dict = {
        f"<e{i}>": tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }

    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]

    return tokenizer, config


def build_or_load_gen_model(args, load_model=True):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Loading from local
    local_dir = '../models/'

    print("Loading config from: ", local_dir+args.model_name_or_path)
    config = config_class.from_pretrained(local_dir+args.model_name_or_path)

    if not args.tokenizer_name:      # default codet5 tokenizer
        tokenizer_name = "Salesforce/codet5-base"

    print("Loading tokenizer from: ", local_dir+args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(local_dir+args.model_name_or_path)

    if not args.model_name_or_path:
        if args.model_type == "codet5":
            args.model_name_or_path = "Salesforce/codet5-base"
        else:
            args.model_name_or_path = "t5-base"

    elif args.from_scratch is True:
        model = model_class(config)
    else:
        print("Loading model from: ", local_dir+args.model_name_or_path)
        model = model_class.from_pretrained(local_dir + args.model_name_or_path)

    tokenizer, config = enrich_vocab(args, tokenizer, config)
    model.config = config  # update the config in model
    model.resize_token_embeddings(len(tokenizer))
    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path
    )

    # model.to(args.device)
    if args.load_model_path is not None and load_model is True:
        logger.info("Load model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(
            args.load_model_path, map_location="cpu"))
        logger.info("Model from {} has been loaded".format(args.load_model_path))
    # model.to(args.device)
    return config, model, tokenizer



MODEL_CLASSES = {
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
}
