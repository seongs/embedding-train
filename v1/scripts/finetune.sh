# 1. 데이터 캐싱
python cache_data.py

# 2. 학습
EPOCH=2
LR=2e-4
BATCH_SIZE=4
DATE=240809

export WANDB_PROJECT="KUKE"
export WANDB_NAME="KUKE-bs=${BATCH_SIZE}-ep=${EPOCH}-lr=${LR}-${DATE}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 train.py \
    --model_name_or_path intfloat/multilingual-e5-large \
    --output_dir /data/ONTHEIT/MODELS/${WANDB_NAME} \
    --data_dir /data/ONTHEIT/DATA/test-240719/ \
    --cache_dir /mnt/raid6/yjoonjang/projects/KoE5/cache-test \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --warmup_steps 100 \
    --logging_steps 2 \
    --save_steps 100 \
    --cl_temperature 0.02 \
    --test False