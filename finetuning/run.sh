CURRENT_PATH=`dirname $0`
cd $CURRENT_PATH

PORT=$1
GPUID=$2
MODEL=$3
MODEL_TAG=$4
TASK=$5
MaxSL=$6
MaxTL=$7
DataDir=$8
BSize=$9
GAcc=${10}
LR=${11}
# MODEL=microsoft/codereviewer
# MODEL_TAG=codereviewer

# MODEL=Salesforce/codet5-base
# MODEL_TAG=codet5

SPLIT=_
SAVE_TAG=$TASK$SPLIT$MODEL_TAG
SAVE_DIR=./saved_models/$SAVE_TAG

CACHE_PATH=$SAVE_DIR/cache_file
LOG=./$SAVE_DIR/log.txt

if [ ! -d "./saved_models" ];then
mkdir ./saved_models
fi
if [ ! -d "$SAVE_DIR" ];then
mkdir $SAVE_DIR
fi

## accelerate training
CUDA_VISIBLE_DEVICES=$GPUID accelerate launch --main_process_port $PORT --num_processes 1 --config_file ./acc_config.yaml train.py \
    --load $MODEL \
    --output_dir $SAVE_DIR \
    --task $TASK \
    --max_source_len $MaxSL \
    --max_target_len $MaxTL \
    --cache_path $CACHE_PATH\
    --data_dir $DataDir \
    --batch_size_per_replica $BSize \
    --grad_acc_steps $GAcc\
    --lr $LR \
    --epochs 20 \
    2>&1 | tee ${LOG}