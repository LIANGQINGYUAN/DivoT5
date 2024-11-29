#!/bin/bash
# NCCL_P2P_DISABLE=1
# NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2
# PER_NODE_GPU=7

CURRENT_DIR=`pwd`
NCCL_DEBUG=INFO

# python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU $SCRIPT_PATH \
# torchrun --standalone --nnodes=1 --nproc_per_node=$PER_NODE_GPU $SCRIPT_PATH \
# small 64, base 48
# base 16
# small 
DTAG=vbase300

function pretrain() {
    SCRIPT_PATH="src/pre_training/pretrain_$DTAG.py"
    ACC_CONFIG="src/pre_training/acc_config.yaml"
    accelerate launch --num_processes 3 --main_process_port 8556  --config_file $ACC_CONFIG $SCRIPT_PATH \
        --model_type codet5 \
        --warmup_steps 500 \
        --learning_rate 3e-4 \
        --num_train_epochs 20 \
        --model_name_or_path Salesforce/codet5-base \
        --tokenizer_name Salesforce/codet5-base \
        --data_dir ${CURRENT_DIR}/Dataset/$DTAG \
        --output_dir ${CURRENT_DIR}/outputs/models/pre-training-$DTAG \
        --always_save_model \
        --train_batch_size 47 \
        --gradient_accumulation_steps 4 \
        --max_source_length 512 \
        --max_target_length 512 \
        --mask_rate 0.15 \
        --save_steps 2000 \
        --log_steps 5 \
        --train_steps 200000 \
        --debug
}
pretrain;