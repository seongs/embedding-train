EPOCH=1
LR=2e-5
BATCH_SIZE=512
MINI_BATCH_SIZE=32
DATE=240826

export WANDB_PROJECT="KUKE"
export WANDB_NAME="KUKE-bs=${BATCH_SIZE}-ep=${EPOCH}-lr=${LR}-${DATE}"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
    --model_name_or_path "intfloat/multilingual-e5-large" \
    --output_dir "/data/yjoonjang/KUKE/${WANDB_NAME}" \
    --data_dir "/data/ONTHEIT/DATA/datasets" \
    --num_train_epochs $EPOCH \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $MINI_BATCH_SIZE \
    --mini_batch_size 16 \
    --warmup_steps 100 \
    --logging_steps 2 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --resume_from_checkpoint False \
    --test False
