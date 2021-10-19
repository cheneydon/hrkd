#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'
MODEL_NAME='meta_bert'

SEQ_LEN=128
EPOCHS=3
BS=8
LR=5e-5
SEED=42

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
MNLI_DIR='./dataset/glue'
AMAZON_REVIEW_DIR='./dataset/amazon_review'


# ------------- Train ------------- #

# MNLI
TRAIN_RATIO=0.9
DEV_RATIO=0.1
EXP_DIR='./exp/train_multi_domain/mnli/'
TASK='mnli'
DOMAIN='telephone'
bash dist_train_multi_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH


# Amazon review
EXP_DIR='./exp/train_multi_domain/amazon_review/'
TASK='amazon_review'
DOMAIN='books'
bash dist_train_multi_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH



# ------------- Test ------------- #

# MNLI
GPU_DEVICES='0,1'
EXP_DIR='./exp/train_multi_domain/mnli/'
TASK='mnli'

DOMAIN='fiction'
RESUME_PATH=$EXP_DIR'best_model_fiction.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='government'
RESUME_PATH=$EXP_DIR'best_model_government.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='slate'
RESUME_PATH=$EXP_DIR'best_model_slate.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='telephone'
RESUME_PATH=$EXP_DIR'best_model_telephone.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='travel'
RESUME_PATH=$EXP_DIR'best_model_travel.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH




# Amazon review
GPU_DEVICES='0,1'
EXP_DIR='./exp/train_multi_domain/amazon_review/'
TASK='amazon_review'

DOMAIN='books'
RESUME_PATH=$EXP_DIR'best_model_books.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='dvd'
RESUME_PATH=$EXP_DIR'best_model_dvd.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='electronics'
RESUME_PATH=$EXP_DIR'best_model_electronics.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH

DOMAIN='kitchen'
RESUME_PATH=$EXP_DIR'best_model_kitchen.bin'
python train_multi_domain.py --do_test --gpu_devices $GPU_DEVICES --val_freq 50 --model_name $MODEL_NAME --lowercase --task $TASK --domain $DOMAIN --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --exp_dir $EXP_DIR --max_seq_len $SEQ_LEN --vocab_path $VOCAB_PATH --pretrain_path $PRETRAIN_PATH --resume_path $RESUME_PATH
