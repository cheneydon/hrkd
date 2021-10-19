#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'
MODEL_NAME='bert_base'

SEQ_LEN=128
EPOCHS=3
BS=8
LR=2e-5
SEED=42

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
MNLI_DIR='./dataset/glue'
AMAZON_REVIEW_DIR='./dataset/amazon_review'


# ---------------- Train ---------------- #

# MNLI
TRAIN_RATIO=0.9
DEV_RATIO=0.1
EXP_DIR='./exp/mnli/train_domain/'
TASK='mnli'
DOMAIN='fiction'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='government'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='slate'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='telephone'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='travel'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH



# Amazon review
EXP_DIR='./exp/amazon_review/'
TASK='amazon_review'
DOMAIN='books'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='dvd'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='electronics'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH
DOMAIN='kitchen'
bash dist_train_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH




# ---------------- Test ---------------- #


# MNLI
GPU_DEVICES='0,1'
TASK='mnli'

EXP_DIR='./exp/train_domain/mnli/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='fiction'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/mnli/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='government'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/mnli/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='slate'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/mnli/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='telephone'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/mnli/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='travel'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH



# Amazon review
GPU_DEVICES='0,1'
TASK='amazon_review'

EXP_DIR='./exp/train_domain/amazon_review/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='books'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/amazon_review/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='dvd'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/amazon_review/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='electronics'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

EXP_DIR='./exp/train_domain/amazon_review/'
RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='kitchen'
python train_domain.py --gpu_devices $GPU_DEVICES --do_test --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH
