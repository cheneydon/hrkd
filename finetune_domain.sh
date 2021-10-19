#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'

SEQ_LEN=128
BS=8
EPOCHS=10
LR=5e-5
HIDDEN_RATIO=1
PRED_RATIO=1
VAL_FREQ=50

STUDENT_MODEL='tiny_bert'
MNLI_DIR='./dataset/glue'
AMAZON_REVIEW_DIR='./dataset/amazon_review'
VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
STUDENT_PRETRAIN_PATH='./pretrained_ckpt/2nd_General_TinyBERT_4L_312D/pytorch_model.bin'


# ----------------- Train ----------------- #

# MNLI
TRAIN_RATIO=0.9
DEV_RATIO=0.1
PRETRAIN_DIR='./exp/train_domain/mnli/'
EXP_DIR='./exp/finetune_domain/mnli/'
TASK='mnli'

DOMAIN='fiction'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/fiction/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='government'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/government/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='slate'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/slate/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='telephone'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/telephone/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='travel'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/travel/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR



# Amazon review
PRETRAIN_DIR='./exp/train_domain/amazon_review/'
EXP_DIR='./exp/finetune_domain/amazon_review/'
TASK='amazon_review'

DOMAIN='books'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/books/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='dvd'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/dvd/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='electronics'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/electronics/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR

DOMAIN='kitchen'
TEACHER_PRETRAIN_PATH=$PRETRAIN_DIR'/kitchen/best_model.bin'
bash dist_finetune_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR




# ----------------- Test ----------------- #

# MNLI
GPU_DEVICES='2,3'
TASK='mnli'
FINETUNE_DIR='./exp/finetune_domain/mnli/'

EXP_DIR=$FINETUNE_DIR'/fiction/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='fiction'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

EXP_DIR=$FINETUNE_DIR'/government/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='government'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

EXP_DIR=$FINETUNE_DIR'/slate/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='slate'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

EXP_DIR=$FINETUNE_DIR'/telephone/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='telephone'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

EXP_DIR=$FINETUNE_DIR'/travel/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
DOMAIN='travel'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR


# Amazon review
TASK='amazon_review'
DOMAIN='books'
EXP_DIR='./exp/finetune_domain/amazon_review/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='dvd'
EXP_DIR='./exp/finetune_domain/amazon_review/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='electronics'
EXP_DIR='./exp/finetune_domain/amazon_review/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='kitchen'
EXP_DIR='./exp/finetune_domain/amazon_review/'
STUDENT_RESUME_PATH=$EXP_DIR'best_model.bin'
python finetune_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --student_model $STUDENT_MODEL --task $TASK  --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $STUDENT_RESUME_PATH --exp_dir $EXP_DIR
