EPOCH=2
LR=2e-5
BATCH_SIZE=32768
MINI_BATCH_SIZE=16
DATE=240904

BATCH_SIZE_DIV8=$((BATCH_SIZE / 4))

export WANDB_PROJECT="KUKE"

# === PT ===
#export WANDB_NAME="KUKE-pt-data=prefinetuning_ko-bs=${BATCH_SIZE_DIV8}-ep=${EPOCH}-lr=${LR}-${DATE}"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
#    --model_name_or_path "intfloat/multilingual-e5-large" \
#    --output_dir "/data/yjoonjang/KUKE/${WANDB_NAME}" \
#    --use_hf_dataset True \
#    --data_dir "nlpai-lab/prefinetuning-embed-full-v1" \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --per_device_train_batch_size $BATCH_SIZE \
#    --per_device_eval_batch_size $BATCH_SIZE \
#    --mini_batch_size $MINI_BATCH_SIZE \
#    --warmup_steps 100 \
#    --logging_steps 1 \
#    --max_seq_length 512 \
#    --save_strategy epoch \
#    --resume_from_checkpoint False \
#    --test False

# === FT === hf_data False
export WANDB_NAME="KUKE-ft_with_openqp_pair_without_hn-loss=CachedMultipleNegativesRankingLoss-bs=${BATCH_SIZE_DIV8}-ep=${EPOCH}-lr=${LR}-${DATE}"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=48020 train.py \
    --model_name_or_path "intfloat/multilingual-e5-large" \
    --output_dir "/data/yjoonjang/KUKE/${WANDB_NAME}" \
    --use_hf_dataset False \
    --data_dir "/data/ONTHEIT/DATA/data_without_ontheit" \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --warmup_steps 100 \
    --logging_steps 1 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --resume_from_checkpoint False \
    --test False